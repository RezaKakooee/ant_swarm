"""Train PPO (SB3) on the AntSwarmBarrier environment.

Usage:
    python train_ppo.py
    python train_ppo.py --timesteps 5_000_000 --n-envs 8
    python train_ppo.py --eval --model storage_local/2026_05_30_1200__ant__ppo/checkpoints/best/best_model
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).parent))
from ant_swarm import AntSwarmEnv  # noqa: E402

PROJECT_ROOT = Path(__file__).parent
STORAGE_DIR  = PROJECT_ROOT / "storage_local"


def _make_run_dir() -> Path:
    """Create and return a timestamped run directory under storage_local."""
    name = datetime.now().strftime("%Y_%m_%d_%H%M") + "__ant__ppo"
    run_dir = STORAGE_DIR / name
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

        anim = FuncAnimation(
            fig, update, frames=len(frames), interval=1000 // self.fps, blit=True,
        )
        anim.save(str(out), writer=PillowWriter(fps=self.fps))
        plt.close(fig)
        print(f"  [render] {out.relative_to(STORAGE_DIR)}  ({len(frames)} frames)", flush=True)


def make_env(seed: int = 0):
    def _init():
        env = AntSwarmEnv(seed=seed)
        env = FlattenObservation(env)
        return env
    return _init


def train(args):
    run_dir  = _make_run_dir()
    ckpt_dir = run_dir / "checkpoints"
    rend_dir = run_dir / "renders"
    tb_dir   = run_dir / "tb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir   : {run_dir}", flush=True)
    print(f"Ckpts     : {ckpt_dir}", flush=True)
    print(f"Renders   : {rend_dir}", flush=True)

    vec_env = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    eval_env = DummyVecEnv([make_env(seed=999)])
    eval_env = VecMonitor(eval_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=args.n_steps,
        batch_size=512,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=0,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(ckpt_dir / "eval_logs"),
            eval_freq=max(20_000 // args.n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        RenderCallback(
            render_freq=max(args.render_freq // args.n_envs, 1),
            save_dir=rend_dir,
            fps=30,
            seed=0,
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    final_path = ckpt_dir / "ppo_final"
    model.save(str(final_path))
    print(f"Saved → {final_path}.zip", flush=True)


def evaluate(args):
    env = FlattenObservation(AntSwarmEnv(seed=42))
    model = PPO.load(args.model, env=env)

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
    p.add_argument("--timesteps", type=int, default=50_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=4096,
                   help="PPO rollout steps per env per update")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--render-freq", type=int, default=500_000,
                   help="save a policy GIF every this many env steps")
    args = p.parse_args()

    if args.eval:
        if args.model is None:
            p.error("--eval requires --model <path>")
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
