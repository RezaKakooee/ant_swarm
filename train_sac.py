"""Train SAC (SB3) on the AntSwarmBarrier environment.

SAC advantages over PPO for this task:
  * Off-policy — far more sample-efficient; learns from every transition
  * Auto entropy tuning — automatically balances exploration vs exploitation
  * Designed for continuous action spaces (our [angle, magnitude] per ant)

Usage:
    python train_sac.py
    python train_sac.py --timesteps 5_000_000 --no-wandb
    python train_sac.py --eval --model storage_local/<run>/checkpoints/best/best_model
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).parent))
from ant_swarm import AntSwarmEnv  # noqa: E402

PROJECT_ROOT = Path(__file__).parent
STORAGE_DIR  = PROJECT_ROOT / "storage_local"

WANDB_PROJECT = "ant_swarm"
WANDB_ENTITY  = "kakooee"


def _make_run_name() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    return f"ant__{ts}__{job_id}__train_sac.py"


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


def train(args):
    run_name = _make_run_name()
    run_dir  = _make_run_dir(run_name)
    ckpt_dir = run_dir / "checkpoints"
    rend_dir = run_dir / "renders"
    tb_dir   = run_dir / "tb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name  : {run_name}", flush=True)
    print(f"Run dir   : {run_dir}", flush=True)

    # --- wandb ---
    wandb_run = None
    if args.wandb:
        import wandb
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            dir=str(STORAGE_DIR),   # wandb/ folder lives under storage_local
            group="sac",
            tags=["sac", "ant_swarm"],
            config={
                "timesteps": args.timesteps,
                "buffer_size": args.buffer_size,
                "batch_size": args.batch_size,
                "learning_starts": args.learning_starts,
                "gamma": 0.999,
                "ent_coef": "auto",
                "learning_rate": 3e-4,
            },
            sync_tensorboard=True,
            save_code=False,
        )
        print(f"W&B run   : {wandb_run.url}", flush=True)

    # SAC works with a single env (off-policy; parallelism via replay buffer, not rollouts)
    env = make_env(seed=0)()
    env = VecMonitor(DummyVecEnv([lambda: env]))

    eval_env = VecMonitor(DummyVecEnv([make_env(seed=999)]))

    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        gamma=0.999,       # high gamma: sparse reward at episode end must propagate far
        tau=0.005,
        ent_coef="auto",   # automatic entropy tuning — key SAC advantage
        learning_rate=3e-4,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=0,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(ckpt_dir),
            name_prefix="sac",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(ckpt_dir / "eval_logs"),
            eval_freq=20_000,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        RenderCallback(
            render_freq=args.render_freq,
            save_dir=rend_dir,
            fps=30,
            seed=0,
        ),
    ]

    if args.wandb:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(gradient_save_freq=0, verbose=0))

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final_path = ckpt_dir / "sac_final"
    model.save(str(final_path))
    print(f"Saved → {final_path}.zip", flush=True)

    if wandb_run is not None:
        wandb_run.finish()


def evaluate(args):
    env = FlattenObservation(AntSwarmEnv(seed=42))
    model = SAC.load(args.model, env=env)

    returns, lengths, successes = [], [], []
    for ep in range(args.eval_episodes):
        obs, _ = env.reset()
        total_r, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done = terminated or truncated
        returns.append(total_r)
        lengths.append(info["step"])
        successes.append(info.get("object_distance", 1.0) < 0.05)
        print(f"  ep {ep+1:3d}  return={total_r:.2f}  steps={info['step']}  "
              f"dist={info.get('object_distance', float('nan')):.3f}")

    print(f"\nmean return : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"mean steps  : {np.mean(lengths):.0f}")
    print(f"success rate: {np.mean(successes)*100:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=5_000_000,
                   help="SAC is sample-efficient; 5M is often enough")
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=10_000,
                   help="fill replay buffer before first gradient step")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--render-freq", type=int, default=100_000,
                   help="save a policy GIF every this many env steps")
    p.add_argument("--no-wandb", dest="wandb", action="store_false",
                   help="disable W&B logging (on by default)")
    args = p.parse_args()

    if args.eval:
        if args.model is None:
            p.error("--eval requires --model <path>")
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
