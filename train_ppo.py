"""Train PPO (SB3) on the AntSwarmBarrier environment.

No CLI args — everything is configured in ``config.yaml`` (`run:` + `ppo:`
sections). To evaluate instead of train, set `run.eval: true` and
`run.eval_model: <checkpoint.zip>`. To warm-start, set `run.init_from`.

    python train_ppo.py
"""
from __future__ import annotations

import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).parent))
from ant_swarm import AntSwarmEnv, load_config, save_code  # noqa: E402
from train_utils import SuccessTrajectoryCallback, CurriculumCallback  # noqa: E402

PROJECT_ROOT = Path(__file__).parent
STORAGE_DIR  = PROJECT_ROOT / "storage_local"

# Defaults if config.yaml lacks the `run:` / `ppo:` sections.
RUN_DEFAULTS = dict(
    wandb=True, render_freq=500_000, init_from=None,
    eval=False, eval_model=None, eval_episodes=20,
)
PPO_DEFAULTS = dict(
    timesteps=50_000_000, n_envs=8,
    n_steps=4096, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, learning_rate=3e-4,
    ent_coef=0.05, ent_coef_final=None, use_sde=False, sde_sample_freq=16,
    log_std_init=0.0,
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
WANDB_ENTITY  = "kakooee"


def _make_run_name(n_ants: int) -> str:
    """Run name: ``ant__YYYYMMDD_HHMM__<jobid>__train_ppo__<single|multi>``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    mode = "single" if n_ants == 1 else "multi"
    return f"ant__{ts}__{job_id}__train_ppo__{mode}"


def _make_run_dir(run_name: str) -> Path:
    run_dir = STORAGE_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class RenderCallback(BaseCallback):
    """Roll out one deterministic episode every ``render_freq`` steps and save a GIF."""

    def __init__(self, render_freq: int, save_dir: str | Path, fps: int = 30, seed: int = 0):
        super().__init__()
        self.render_freq = render_freq
        self.save_dir = Path(save_dir)
        self.fps = fps
        self.seed = seed

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self._save_gif()
        return True

    def _save_gif(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        env = AntSwarmEnv(seed=self.seed)
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
        print(f"  [render] {out.relative_to(STORAGE_DIR)}  ({len(frames)} frames)", flush=True)

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


def make_env(seed: int = 0):
    def _init():
        env = AntSwarmEnv(seed=seed)
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
    save_code(run_dir, __file__)   # snapshot code + config for reproducibility

    print(f"Run name  : {run_name}", flush=True)
    print(f"Run dir   : {run_dir}", flush=True)
    print(f"Exploration: ent_coef={s['ent_coef']} anneal->{s['ent_coef_final']} "
          f"use_sde={s['use_sde']} log_std_init={s['log_std_init']}", flush=True)

    # --- wandb ---
    wandb_run = None
    if s["wandb"]:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
            dir=str(run_dir), group="ppo", tags=["ppo", "ant_swarm"],
            config=s, sync_tensorboard=True, save_code=False,
        )
        print(f"W&B run   : {wandb_run.url}", flush=True)

    vec_env = VecMonitor(DummyVecEnv([make_env(i) for i in range(s["n_envs"])]))
    eval_env = VecMonitor(DummyVecEnv([make_env(seed=999)]))

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
        print(f"Warm-started from: {s['init_from']}", flush=True)

    reach = cfg.goal.reach_radius
    callbacks = [
        CheckpointCallback(save_freq=max(50_000 // s["n_envs"], 1),
                           save_path=str(ckpt_dir), name_prefix="ppo"),
        EvalCallback(eval_env, best_model_save_path=str(ckpt_dir / "best"),
                     log_path=str(ckpt_dir / "eval_logs"),
                     eval_freq=max(20_000 // s["n_envs"], 1),
                     n_eval_episodes=10, deterministic=True, verbose=1),
        RenderCallback(render_freq=max(s["render_freq"] // s["n_envs"], 1),
                       save_dir=rend_dir, fps=30, seed=0),
        EpisodeMetricsCallback(reach_radius=reach),
        SuccessTrajectoryCallback(save_dir=run_dir / "successes", reach_radius=reach),
    ]

    # optional entropy-coefficient annealing
    if s["ent_coef_final"] is not None:
        callbacks.append(EntCoefAnneal(start=s["ent_coef"], final=s["ent_coef_final"],
                                       total_timesteps=s["timesteps"]))

    # optional gap-size curriculum (eval held at the hard target)
    cur = getattr(cfg, "curriculum", None)
    if cur is not None and getattr(cur, "enabled", False):
        callbacks.append(CurriculumCallback(
            start=cur.start_wall_len, target=cur.target_wall_len, step=cur.step,
            success_threshold=cur.success_threshold, window=cur.window, reach_radius=reach,
            max_steps_per_stage=getattr(cur, "max_steps_per_stage", None),
            stop_on_master=getattr(cur, "stop_on_master", False),
            stop_success=getattr(cur, "stop_success", 0.9),
            stop_window=getattr(cur, "stop_window", 200),
        ))
        eval_env.env_method("set_wall_length", cur.target_wall_len)
        print(f"Curriculum: wall_len {cur.start_wall_len} -> {cur.target_wall_len} "
              f"(eval fixed at {cur.target_wall_len})", flush=True)

    if s["wandb"]:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(gradient_save_freq=0, verbose=0))

    model.learn(total_timesteps=s["timesteps"], callback=callbacks)

    final_path = ckpt_dir / "ppo_final"
    model.save(str(final_path))
    print(f"Saved → {final_path}.zip", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


def evaluate(cfg, s):
    if not s["eval_model"]:
        raise SystemExit("Set run.eval_model in config.yaml to a checkpoint .zip")
    env = FlattenObservation(AntSwarmEnv(seed=42))
    model = PPO.load(s["eval_model"], env=env)
    reach = cfg.goal.reach_radius

    returns, lengths, successes = [], [], []
    for ep in range(s["eval_episodes"]):
        obs, _ = env.reset()
        total_r, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done = terminated or truncated
        returns.append(total_r)
        lengths.append(info["step"])
        successes.append(info.get("object_distance", 1.0) < reach)
        print(f"  ep {ep+1:3d}  return={total_r:.2f}  steps={info['step']}  "
              f"dist={info.get('object_distance', float('nan')):.3f}")

    print(f"\nmean return : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"mean steps  : {np.mean(lengths):.0f}")
    print(f"success rate: {np.mean(successes)*100:.1f}%")


def main():
    cfg = load_config()
    s = _settings(cfg)
    if s["eval"]:
        evaluate(cfg, s)
    else:
        train(cfg, s)


if __name__ == "__main__":
    main()
