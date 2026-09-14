"""Train PPO (SB3) on the AntSwarmBarrier environment.

No CLI args — everything is configured in ``config.yaml`` (`run:` + `ppo:`
sections). To evaluate instead of train, set `run.eval: true` and
`run.eval_model: <checkpoint.zip>`. To warm-start, set `run.init_from`.

    python scripts/rl/train_ppo.py [--config-name pnas_kin_geo] [key=value ...]
"""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ant_swarm import (AntSwarmEnv, build_run_id, load_config_cli,  # noqa: E402
                       save_code, setup_logging)
from exploration import LogStdClampCallback  # noqa: E402
from train_utils import (SuccessTrajectoryCallback, build_curriculum,  # noqa: E402
                         pin_eval_hard, pin_standalone_eval_hard,
                         prepare_curriculum)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR  = PROJECT_ROOT / "storage_local"

# Defaults if config.yaml lacks the `run:` / `ppo:` sections.
RUN_DEFAULTS = dict(
    wandb=True, render_freq=500_000, init_from=None,
    eval=False, eval_model=None, eval_episodes=20,
    eval_render=True, eval_render_dir=None, eval_fps=30, eval_deterministic=True,
    save_successes=True, dedup_successes=True, dedup_tol=0.05,
)
PPO_DEFAULTS = dict(
    timesteps=50_000_000, n_envs=8,
    n_steps=4096, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, learning_rate=3e-4,
    ent_coef=0.001, ent_coef_final=0.0, use_sde=False, sde_sample_freq=16,
    log_std_init=-1.0, log_std_min=-3.0, log_std_max=-0.5,
)


def _settings(cfg) -> dict:
    """Merge the `run:` + `ppo:` config sections into one dict (with fallbacks)."""
    run = getattr(cfg, "run", None)
    ppo = getattr(cfg, "ppo", None)
    s = {k: getattr(run, k, d) for k, d in RUN_DEFAULTS.items()}
    s.update({k: getattr(ppo, k, d) for k, d in PPO_DEFAULTS.items()})
    return s


class EntCoefAnneal(BaseCallback):
    """Linearly anneal PPO's entropy coefficient from `start` to `final`."""

    def __init__(self, start: float, final: float, total_timesteps: int):
        super().__init__()
        self.start, self.final, self.total = start, final, total_timesteps

    def _on_step(self) -> bool:
        frac = min(self.num_timesteps / max(self.total, 1), 1.0)
        self.model.ent_coef = self.start + frac * (self.final - self.start)
        return True


class EpisodeMetricsCallback(BaseCallback):
    """Log success rate + final distance-to-goal over a rolling window of episodes.

    (Reward and length are already logged by VecMonitor as rollout/ep_rew_mean
    and rollout/ep_len_mean; this adds the task-specific signals.)
    """

    def __init__(self, reach_radius: float, window: int = 100):
        super().__init__()
        self.reach_radius = reach_radius
        self.success = deque(maxlen=window)
        self.final_dist = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done:
                d = info.get("object_distance")
                if d is not None:
                    self.final_dist.append(d)
                    self.success.append(float(d < self.reach_radius))
        if self.success:
            self.logger.record("rollout/success_rate", sum(self.success) / len(self.success))
            self.logger.record("rollout/final_dist_mean", sum(self.final_dist) / len(self.final_dist))
        return True

WANDB_PROJECT = "ant_swarm"
WANDB_ENTITY  = os.environ.get("ANT_SWARM_WANDB_ENTITY") or os.environ.get("WANDB_ENTITY") or None


def _make_run_name(n_ants: int) -> str:
    """One id shared by run dir, wandb, and the ops log (see ant_swarm/run_id.py)."""
    return build_run_id("train_ppo", n_ants)


def _make_run_dir(run_name: str) -> Path:
    run_dir = STORAGE_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class RenderCallback(BaseCallback):
    """Roll out one deterministic episode every ``render_freq`` steps and save a GIF."""

    def __init__(self, cfg, render_freq: int, save_dir: str | Path, fps: int = 30,
                 seed: int = 0, pose=None, wall_len=None):
        super().__init__()
        self.cfg = cfg
        self.render_freq = render_freq
        self.save_dir = Path(save_dir)
        self.fps = fps
        self.seed = seed
        self.pose = None if pose is None else np.asarray(pose, dtype=np.float32)
        self.wall_len = None if wall_len is None else float(wall_len)

    def pin_pose(self, pose, wall_len=None) -> None:
        """Render every policy check from one canonical hard pose."""
        self.pose = np.asarray(pose, dtype=np.float32)
        self.wall_len = None if wall_len is None else float(wall_len)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self._save_gif()
        return True

    def _save_gif(self):
        env = AntSwarmEnv(config=self.cfg, seed=self.seed)
        if self.wall_len is not None:
            env.set_wall_length(self.wall_len)
        if self.pose is not None:
            env.set_spawn_pose(self.pose)
        flat_env = FlattenObservation(env)
        # fixed seed only for pinned (curriculum) scenes; vary otherwise so
        # random-start / random-goal runs show a fresh episode each snapshot
        seed = self.seed if self.pose is not None \
            else self.seed + self.num_timesteps
        obs, _ = flat_env.reset(seed=seed)

        frames = [env.render()]
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = flat_env.step(action)
            frames.append(env.render())
            done = terminated or truncated

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out = self.save_dir / f"policy_{self.num_timesteps:08d}.gif"

        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(str(out), save_all=True, append_images=imgs[1:], loop=0,
                     duration=max(1, int(1000 / self.fps)), optimize=True)
        logger.info(f"[render] {out.relative_to(STORAGE_DIR)}  ({len(frames)} frames)")

        # --- log to TensorBoard ---
        try:
            from stable_baselines3.common.logger import TensorBoardOutputFormat
            for fmt in self.logger.output_formats:
                if isinstance(fmt, TensorBoardOutputFormat):
                    # SummaryWriter.add_video expects (N, T, C, H, W) uint8
                    vid = np.stack(frames)[None].transpose(0, 1, 4, 2, 3)
                    fmt.writer.add_video("render/policy", vid,
                                         global_step=self.num_timesteps, fps=self.fps)
                    fmt.writer.flush()
                    break
        except Exception:
            pass

        # --- log to W&B ---
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {"render/policy": wandb.Video(str(out), fps=self.fps, format="gif")},
                    step=self.num_timesteps,
                )
        except Exception:
            pass


def make_env(cfg, seed: int = 0, training: bool = False):
    def _init():
        env = AntSwarmEnv(config=cfg, seed=seed)
        icfg = getattr(cfg.env, "intrinsic", None)
        if training and icfg is not None and bool(getattr(icfg, "enabled", False)):
            from intrinsic import IntrinsicRewardWrapper
            env = IntrinsicRewardWrapper(env, icfg)
        env = FlattenObservation(env)
        return env
    return _init


def train(cfg, s):
    run_name = _make_run_name(int(cfg.ants.n))
    run_dir  = _make_run_dir(run_name)
    ckpt_dir = run_dir / "checkpoints"
    rend_dir = run_dir / "renders"
    tb_dir   = run_dir / "tb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)         # console + <run_dir>/train.log
    save_code(run_dir, __file__, cfg=cfg)   # snapshot code + resolved config

    logger.info(f"Run name  : {run_name}")
    logger.info(f"Run dir   : {run_dir}")
    logger.info(f"Exploration: ent_coef={s['ent_coef']} anneal->{s['ent_coef_final']} "
                f"use_sde={s['use_sde']} log_std_init={s['log_std_init']} "
                f"log_std_bounds=[{s['log_std_min']}, {s['log_std_max']}]")

    # --- wandb ---
    wandb_run = None
    if s["wandb"]:
      try:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
            group="ppo", tags=["ppo", "ant_swarm"],
            config=s, sync_tensorboard=True, save_code=False,
        )
        logger.info(f"W&B run   : {wandb_run.url}")
      except Exception as e:
        logger.warning(f"wandb init failed — continuing without tracking: {e}")
        s["wandb"] = False

    vec_env = VecMonitor(DummyVecEnv([make_env(cfg, seed=i, training=True) for i in range(s["n_envs"])]))
    eval_env = VecMonitor(DummyVecEnv([make_env(cfg, seed=999)]))

    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=s["n_steps"], batch_size=s["batch_size"], n_epochs=s["n_epochs"],
        gamma=s["gamma"], gae_lambda=s["gae_lambda"], clip_range=s["clip_range"],
        ent_coef=s["ent_coef"], learning_rate=s["learning_rate"],
        use_sde=s["use_sde"], sde_sample_freq=s["sde_sample_freq"],
        policy_kwargs=dict(log_std_init=s["log_std_init"]),
        verbose=1, tensorboard_log=str(tb_dir), seed=0,
    )

    if s["init_from"]:   # warm-start weights from an existing checkpoint
        model.set_parameters(s["init_from"])
        logger.info(f"Warm-started from: {s['init_from']}")

    reach = cfg.goal.reach_radius
    render_callback = RenderCallback(
        cfg=cfg, render_freq=max(s["render_freq"] // s["n_envs"], 1),
        save_dir=rend_dir, fps=30, seed=0)
    callbacks = [
        CheckpointCallback(save_freq=max(50_000 // s["n_envs"], 1),
                           save_path=str(ckpt_dir), name_prefix="ppo"),
        EvalCallback(eval_env, best_model_save_path=str(ckpt_dir / "best"),
                     log_path=str(ckpt_dir / "eval_logs"),
                     eval_freq=max(20_000 // s["n_envs"], 1),
                     n_eval_episodes=10, deterministic=True, verbose=1),
        render_callback,
        EpisodeMetricsCallback(reach_radius=reach),
    ]
    if s["log_std_min"] is not None or s["log_std_max"] is not None:
        callbacks.append(LogStdClampCallback(
            min_log_std=s["log_std_min"], max_log_std=s["log_std_max"]))
    if s["save_successes"]:
        callbacks.append(SuccessTrajectoryCallback(
            save_dir=run_dir / "successes", reach_radius=reach,
            dedup=s["dedup_successes"], dedup_tol=s["dedup_tol"],
        ))

    # optional entropy-coefficient annealing
    if s["ent_coef_final"] is not None:
        callbacks.append(EntCoefAnneal(start=s["ent_coef"], final=s["ent_coef_final"],
                                       total_timesteps=s["timesteps"]))

    # optional curriculum (gap or reverse); eval held at the real hard task
    cur = getattr(cfg, "curriculum", None)
    if cur is not None and getattr(cur, "enabled", False):
        curriculum = build_curriculum(cur, reach, cfg=cfg)
        prepare_curriculum(vec_env, curriculum)
        if getattr(cur, "mode", "gap") == "pose_path":
            render_callback.pin_pose(curriculum.anchors[-1], curriculum.wall_len)
        callbacks.append(curriculum)
        pin_eval_hard(eval_env, cur, curriculum)
        logger.info(f"Curriculum mode: {getattr(cur, 'mode', 'gap')}  "
                    f"(reward_mode={getattr(cfg.env, 'reward_mode', 'shaped')})")

    if s["wandb"]:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(gradient_save_freq=0, verbose=0))

    model.learn(total_timesteps=s["timesteps"], callback=callbacks)

    final_path = ckpt_dir / "ppo_final"
    model.save(str(final_path))
    logger.info(f"Saved → {final_path}.zip")
    if wandb_run is not None:
        wandb_run.finish()


def _resolve_eval_model_path(path_str: str | Path) -> tuple[Path, Path]:
    """Resolve checkpoint path and experiment directory."""
    path = Path(str(path_str)).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if path.is_dir():
        exp_dir = path
        candidates = [
            path / "checkpoints" / "best" / "best_model.zip",
            path / "checkpoints" / "ppo_final.zip",
            path / "best" / "best_model.zip",
            path / "best_model.zip",
            path / "ppo_final.zip",
        ]
        for c in candidates:
            if c.is_file():
                return c, exp_dir
        zips = sorted((path / "checkpoints").glob("*.zip"))
        if zips:
            return zips[-1], exp_dir
        raise FileNotFoundError(f"No checkpoint .zip found in {path}")
    
    exp_dir = path.parent
    if exp_dir.name in ("best", "eval_logs"):
        exp_dir = exp_dir.parent
    if exp_dir.name == "checkpoints":
        exp_dir = exp_dir.parent
    return path, exp_dir


def _save_eval_gif(frames: list[np.ndarray], out_path: Path, fps: int = 30):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(str(out_path), save_all=True, append_images=imgs[1:], loop=0,
                 duration=max(1, int(1000 / fps)), optimize=True)


def _save_eval_mp4(frames: list[np.ndarray], out_path: Path, fps: int = 30):
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
    except Exception as e:
        logger.warning(f"Could not save MP4: {e}")


def evaluate(cfg, s):
    if not s["eval_model"]:
        raise SystemExit("Set run.eval_model in config.yaml to a checkpoint .zip or experiment dir")
    
    model_file, exp_dir = _resolve_eval_model_path(s["eval_model"])
    logger.info(f"Evaluating checkpoint: {model_file}")
    logger.info(f"Experiment dir       : {exp_dir}")

    # Use run's snapshot config if available to guarantee matching geometry
    snap_cfg_path = exp_dir / "code" / "config.yaml"
    if snap_cfg_path.is_file():
        from ant_swarm import load_config
        logger.info(f"Using snapshot config from: {snap_cfg_path}")
        cfg = load_config(snap_cfg_path)

    reach = cfg.goal.reach_radius
    raw_env = AntSwarmEnv(config=cfg, seed=42)
    cur = getattr(cfg, "curriculum", None)
    if cur is not None and getattr(cur, "enabled", False):
        curriculum = build_curriculum(cur, reach, cfg=cfg)
        pin_standalone_eval_hard(raw_env, cur, curriculum)
    env = FlattenObservation(raw_env)
    model = PPO.load(str(model_file), env=env)

    render_eval = bool(s.get("eval_render", True))
    eval_dir = Path(s["eval_render_dir"]) if s.get("eval_render_dir") else (exp_dir / "eval")
    if render_eval:
        eval_dir.mkdir(parents=True, exist_ok=True)

    fps = int(s.get("eval_fps", 30))
    det = bool(s.get("eval_deterministic", True))
    n_episodes = int(s.get("eval_episodes", 5))

    returns, lengths, successes = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        frames = [raw_env.render()] if render_eval else []
        total_r, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=det)
            obs, reward, terminated, truncated, info = env.step(action)
            if render_eval:
                frames.append(raw_env.render())
            total_r += reward
            done = terminated or truncated
        
        step_count = info.get("step", len(frames) - 1 if render_eval else 0)
        final_d = info.get("object_distance", float("nan"))
        succ = bool(final_d < reach)
        returns.append(total_r)
        lengths.append(step_count)
        successes.append(succ)

        log_msg = f"  ep {ep+1:3d}  return={total_r:.2f}  steps={step_count}  dist={final_d:.3f}  success={succ}"
        print(log_msg)
        logger.info(log_msg)

        if render_eval and frames:
            gif_out = eval_dir / f"eval_ep{ep+1:02d}_len{step_count}.gif"
            mp4_out = eval_dir / f"eval_ep{ep+1:02d}_len{step_count}.mp4"
            _save_eval_gif(frames, gif_out, fps=fps)
            _save_eval_mp4(frames, mp4_out, fps=fps)
            logger.info(f"       Saved video → {gif_out} ({len(frames)} frames)")

    summary_str = (
        f"\nEvaluation summary ({n_episodes} episodes):\n"
        f"  mean return : {np.mean(returns):.2f} ± {np.std(returns):.2f}\n"
        f"  mean steps  : {np.mean(lengths):.0f}\n"
        f"  success rate: {np.mean(successes)*100:.1f}%\n"
    )
    if render_eval:
        summary_str += f"  videos dir  : {eval_dir}\n"
    print(summary_str)
    logger.info(summary_str)


def main():
    setup_logging()
    cfg = load_config_cli()
    s = _settings(cfg)
    if s["eval"]:
        evaluate(cfg, s)
    else:
        train(cfg, s)


if __name__ == "__main__":
    main()
