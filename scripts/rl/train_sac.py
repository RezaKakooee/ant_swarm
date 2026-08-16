"""Train SAC (SB3) on the AntSwarmBarrier environment.

SAC advantages over PPO for this task:
  * Off-policy — far more sample-efficient; learns from every transition
  * Auto entropy tuning — automatically balances exploration vs exploitation
  * Designed for continuous action spaces

No CLI args — everything is configured in ``config.yaml`` (`run:` + `sac:`
sections). To evaluate instead of train, set `run.eval: true` and
`run.eval_model: <checkpoint.zip>`. ``run.resume_model`` +
``run.resume_replay`` continue the same reward phase; ``run.transfer_actor_from``
starts a new reward phase with only the learned actor.

    python scripts/rl/train_sac.py [--config-name pnas_kin_geo] [key=value ...]
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from omegaconf import OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ant_swarm import (AntSwarmEnv, build_run_id, load_config_cli,  # noqa: E402
                       save_code, setup_logging)
from train_utils import (SuccessTrajectoryCallback, build_curriculum,  # noqa: E402
                         pin_eval_hard, pin_standalone_eval_hard,
                         prepare_curriculum)
from success_replay_buffer import (SuccessReplayBuffer,  # noqa: E402
                                   seed_success_trajectories)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR  = PROJECT_ROOT / "storage_local"

WANDB_PROJECT = "ant_swarm"
WANDB_ENTITY  = "kakooee"

# Defaults if config.yaml lacks the `run:` / `sac:` sections.
RUN_DEFAULTS = dict(
    wandb=True, render_freq=100_000, init_from=None,
    eval=False, eval_model=None, eval_episodes=20,
    eval_render=True, eval_render_dir=None, eval_fps=30, eval_deterministic=True,
    save_successes=True, dedup_successes=True, dedup_tol=0.05,
    resume_model=None, resume_replay=None, transfer_actor_from=None,
    seed_successes_from=None, seed_successes_limit=None,
    replay_checkpoint_freq=500_000,
)
SAC_DEFAULTS = dict(
    timesteps=15_000_000, buffer_size=1_000_000, batch_size=256,
    learning_starts=10_000, gamma=0.99, tau=0.005, learning_rate=3e-4,
    ent_coef="auto", success_buffer_size=200_000,
    success_batch_fraction=0.25, target_entropy="auto",
)


def _settings(cfg) -> dict:
    """Merge the `run:` + `sac:` config sections into one dict (with fallbacks)."""
    run = getattr(cfg, "run", None)
    sac = getattr(cfg, "sac", None)
    s = {k: getattr(run, k, d) for k, d in RUN_DEFAULTS.items()}
    s.update({k: getattr(sac, k, d) for k, d in SAC_DEFAULTS.items()})
    return s


def _make_run_name(n_ants: int) -> str:
    """One id shared by run dir, wandb, and the ops log (see ant_swarm/run_id.py)."""
    return build_run_id("train_sac", n_ants)


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
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        env = AntSwarmEnv(config=self.cfg, seed=self.seed)
        if self.wall_len is not None:
            env.set_wall_length(self.wall_len)
        if self.pose is not None:
            env.set_spawn_pose(self.pose)
        flat_env = FlattenObservation(env)
        obs, _ = flat_env.reset(seed=self.seed)

        frames = [env.render()]
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = flat_env.step(action)
            frames.append(env.render())
            done = terminated or truncated

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out = self.save_dir / f"policy_{self.num_timesteps:08d}.gif"

        fig, ax = plt.subplots(figsize=(6.5, 4.7))
        im = ax.imshow(frames[0])
        ax.set_axis_off()

        def update(i):
            im.set_data(frames[i])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // self.fps, blit=True)
        anim.save(str(out), writer=PillowWriter(fps=self.fps))
        plt.close(fig)
        logger.info(f"[render] {out.relative_to(STORAGE_DIR)}  ({len(frames)} frames)")

        # --- log to TensorBoard ---
        try:
            from stable_baselines3.common.logger import TensorBoardOutputFormat
            for fmt in self.logger.output_formats:
                if isinstance(fmt, TensorBoardOutputFormat):
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


class EpisodeMetricsCallback(BaseCallback):
    """Log success rate + final distance-to-goal over a rolling window of episodes."""

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


def make_env(cfg, seed: int = 0):
    def _init():
        env = AntSwarmEnv(config=cfg, seed=seed)
        env = FlattenObservation(env)
        return env
    return _init


def _resume_signature(cfg) -> dict:
    """Reward/dynamics fields that make replay and critics phase-specific."""
    def plain(node):
        return OmegaConf.to_container(node, resolve=True)

    return {
        "reward_mode": str(getattr(cfg.env, "reward_mode", "shaped")),
        "observe_linear_velocity": getattr(
            cfg.env, "observe_linear_velocity", None
        ),
        "goal_track": str(getattr(cfg.env, "goal_track", "center")),
        "env": plain(cfg.env),
        "max_steps": int(cfg.env.max_steps),
        "scene_scale": float(cfg.scene_scale),
        "world": plain(cfg.world),
        "walls": plain(cfg.walls),
        "tshape": plain(cfg.tshape),
        "goal": plain(cfg.goal),
        "ants": plain(cfg.ants),
        "motion": plain(cfg.motion),
        "physics": plain(cfg.physics),
    }


def _snapshot_config_for_checkpoint(checkpoint) -> Path | None:
    path = Path(str(checkpoint)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    for parent in (path.parent, *path.parents):
        candidate = parent / "code" / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def _validate_exact_resume(checkpoint, cfg) -> None:
    snapshot = _snapshot_config_for_checkpoint(checkpoint)
    if snapshot is None:
        raise ValueError(
            f"cannot verify exact resume phase for {checkpoint}: no "
            "code/config.yaml snapshot found; use run.transfer_actor_from instead"
        )
    source_cfg = OmegaConf.load(snapshot)
    source = _resume_signature(source_cfg)
    target = _resume_signature(cfg)
    mismatches = [key for key in target if source.get(key) != target[key]]
    if mismatches:
        details = ", ".join(mismatches)
        raise ValueError(
            f"unsafe exact resume from {checkpoint}: phase/config mismatch in "
            f"{details}. Use run.transfer_actor_from for a reward-phase change."
        )


def _build_sac_model(env, cfg, s, tb_dir: Path, reach_radius: float):
    """Create, resume, or actor-transfer a SAC model with success replay."""
    resume_model = s["resume_model"]
    resume_replay = s["resume_replay"]
    transfer_actor = s["transfer_actor_from"]

    # Backward-compatible meaning for the old "warm-start" option: policy
    # transfer, not a partial optimizer/model resume.
    if s["init_from"]:
        if transfer_actor and str(transfer_actor) != str(s["init_from"]):
            raise ValueError("run.init_from and run.transfer_actor_from disagree")
        transfer_actor = transfer_actor or s["init_from"]
        logger.warning(
            "run.init_from is deprecated; treating it as actor-only transfer. "
            "Use run.resume_model for an exact same-phase continuation."
        )

    if resume_model and transfer_actor:
        raise ValueError(
            "run.resume_model and run.transfer_actor_from are mutually exclusive"
        )
    if resume_replay and not resume_model:
        raise ValueError("run.resume_replay requires run.resume_model")

    replay_kwargs = {
        "success_buffer_size": int(s["success_buffer_size"]),
        "success_batch_fraction": float(s["success_batch_fraction"]),
        "reach_radius": float(reach_radius),
    }

    if resume_model:
        _validate_exact_resume(resume_model, cfg)
        model = SAC.load(
            resume_model,
            env=env,
            tensorboard_log=str(tb_dir),
            replay_buffer_class=SuccessReplayBuffer,
            replay_buffer_kwargs=replay_kwargs,
        )
        if resume_replay:
            model.load_replay_buffer(resume_replay)
            if not isinstance(model.replay_buffer, SuccessReplayBuffer):
                raise TypeError(
                    "resume replay predates SuccessReplayBuffer; resume the model "
                    "without that replay or seed saved success JSONs instead"
                )
            # SB3 updates only the outer buffer device after unpickling.
            model.replay_buffer.set_device(model.device)
            model.replay_buffer.clear_pending_episodes()
            logger.info(
                f"Resumed replay: {resume_replay} "
                f"(regular={model.replay_buffer.size()}, "
                f"success={model.replay_buffer.success_size})"
            )
        logger.info(
            f"Resumed SAC model: {resume_model} "
            f"(timesteps={model.num_timesteps})"
        )
        return model, True

    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=s["buffer_size"],
        batch_size=s["batch_size"],
        learning_starts=s["learning_starts"],
        gamma=s["gamma"],
        tau=s["tau"],
        ent_coef=s["ent_coef"],
        target_entropy=s["target_entropy"],
        learning_rate=s["learning_rate"],
        train_freq=1,
        gradient_steps=1,
        replay_buffer_class=SuccessReplayBuffer,
        replay_buffer_kwargs=replay_kwargs,
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=0,
    )

    if transfer_actor:
        # Load a tiny temporary source model. Copying only the actor deliberately
        # leaves reward-specific critics, target critics, optimizers and entropy
        # state fresh for the new phase.
        source = SAC.load(
            transfer_actor,
            env=env,
            device=model.device,
            buffer_size=1,
            replay_buffer_class=None,
            replay_buffer_kwargs={},
        )
        model.actor.load_state_dict(source.actor.state_dict(), strict=True)
        del source
        logger.info(f"Transferred actor only from: {transfer_actor}")

    return model, False


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

    # --- wandb ---
    wandb_run = None
    if s["wandb"]:
      try:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
            dir=str(run_dir), group="sac", tags=["sac", "ant_swarm"],
            config=s, sync_tensorboard=True, save_code=False,
        )
        logger.info(f"W&B run   : {wandb_run.url}")
      except Exception as e:
        logger.warning(f"wandb init failed — continuing without tracking: {e}")
        s["wandb"] = False

    # SAC works with a single env (off-policy; parallelism via replay buffer, not rollouts)
    env = VecMonitor(DummyVecEnv([make_env(cfg, seed=0)]))
    eval_env = VecMonitor(DummyVecEnv([make_env(cfg, seed=999)]))

    reach = float(cfg.goal.reach_radius)
    model, resumed = _build_sac_model(env, cfg, s, tb_dir, reach)

    if s["seed_successes_from"]:
        if not isinstance(model.replay_buffer, SuccessReplayBuffer):
            raise TypeError("success trajectory seeding requires SuccessReplayBuffer")
        limit = s["seed_successes_limit"]
        stats = seed_success_trajectories(
            model.replay_buffer,
            cfg,
            s["seed_successes_from"],
            max_trajectories=None if limit is None else int(limit),
        )
        if stats["loaded"]:
            # A transferred actor plus verified replay should train immediately,
            # not take SB3's random-action warm-up again.
            model.learning_starts = 0
        logger.info(
            f"Seeded success replay from {s['seed_successes_from']}: "
            f"{stats['loaded']} episodes / {stats['transitions']} transitions "
            f"({stats['skipped']} skipped)"
        )
    render_callback = RenderCallback(
        cfg=cfg, render_freq=s["render_freq"], save_dir=rend_dir, fps=30, seed=0)
    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path=str(ckpt_dir), name_prefix="sac"),
        EvalCallback(eval_env, best_model_save_path=str(ckpt_dir / "best"),
                     log_path=str(ckpt_dir / "eval_logs"), eval_freq=20_000,
                     n_eval_episodes=10, deterministic=True, verbose=1),
        render_callback,
        EpisodeMetricsCallback(reach_radius=reach),
    ]

    replay_checkpoint_freq = s["replay_checkpoint_freq"]
    if replay_checkpoint_freq is not None and int(replay_checkpoint_freq) > 0:
        callbacks.append(CheckpointCallback(
            save_freq=int(replay_checkpoint_freq),
            save_path=str(ckpt_dir),
            name_prefix="sac_resume",
            save_replay_buffer=True,
            verbose=1,
        ))
    if s["save_successes"]:
        callbacks.append(SuccessTrajectoryCallback(
            save_dir=run_dir / "successes", reach_radius=reach,
            dedup=s["dedup_successes"], dedup_tol=s["dedup_tol"],
        ))

    # optional curriculum (gap or reverse); eval held at the real hard task
    cur = getattr(cfg, "curriculum", None)
    if cur is not None and getattr(cur, "enabled", False):
        curriculum = build_curriculum(cur, reach, cfg=cfg)
        prepare_curriculum(env, curriculum)
        if getattr(cur, "mode", "gap") == "pose_path":
            render_callback.pin_pose(curriculum.anchors[-1], curriculum.wall_len)
        callbacks.append(curriculum)
        pin_eval_hard(eval_env, cur, curriculum)
        logger.info(f"Curriculum mode: {getattr(cur, 'mode', 'gap')}  "
                    f"(reward_mode={getattr(cfg.env, 'reward_mode', 'shaped')})")

    if s["wandb"]:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(gradient_save_freq=0, verbose=0))

    model.learn(
        total_timesteps=s["timesteps"],
        callback=callbacks,
        reset_num_timesteps=not resumed,
    )

    final_path = ckpt_dir / "sac_final"
    replay_path = ckpt_dir / "sac_final_replay_buffer.pkl"
    model.save(str(final_path))
    model.save_replay_buffer(replay_path)
    logger.info(f"Saved → {final_path}.zip")
    logger.info(f"Saved → {replay_path}")
    if isinstance(model.replay_buffer, SuccessReplayBuffer):
        logger.info(
            f"Final replay: regular={model.replay_buffer.size()}, "
            f"success={model.replay_buffer.success_size} transitions"
        )
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
            path / "checkpoints" / "sac_final.zip",
            path / "best" / "best_model.zip",
            path / "best_model.zip",
            path / "sac_final.zip",
        ]
        for c in candidates:
            if c.is_file():
                return c, exp_dir
        # fallback: any zip in checkpoints/
        zips = sorted((path / "checkpoints").glob("*.zip"))
        if zips:
            return zips[-1], exp_dir
        raise FileNotFoundError(f"No checkpoint .zip found in {path}")
    
    # Path is a file
    exp_dir = path.parent
    if exp_dir.name in ("best", "eval_logs"):
        exp_dir = exp_dir.parent
    if exp_dir.name == "checkpoints":
        exp_dir = exp_dir.parent
    return path, exp_dir


def _save_eval_gif(frames: list[np.ndarray], out_path: Path, fps: int = 30):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    h, w = frames[0].shape[:2]
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * h / w))
    im = ax.imshow(frames[0])
    ax.set_axis_off()
    anim = FuncAnimation(fig, lambda i: [im.set_data(frames[i]) or im],
                         frames=len(frames), interval=1000 / fps, blit=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)


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
    model = SAC.load(str(model_file), env=env)

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
