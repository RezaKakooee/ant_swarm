"""Random agent for the AntSwarmBarrier env — samples actions uniformly.

A baseline / sanity-check policy: every step each ant gets a random
``[angle, magnitude]`` from the action space.

Saves one GIF per episode by default into a timestamped run dir under
``storage_local/<date>__ant__rnd/renders/``.

Usage:
    python random_agent.py                      # 5 episodes, one GIF each
    python random_agent.py --episodes 20
    python random_agent.py --no-gif             # stats only, no GIFs
    python random_agent.py --episodes 3 --seed 3
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ant_swarm import AntSwarmEnv  # noqa: E402

PROJECT_ROOT = Path(__file__).parent
STORAGE_DIR = PROJECT_ROOT / "storage_local"


def _make_run_dir() -> Path:
    """Timestamped run dir under storage_local, mirroring train_ppo ('rnd' tag)."""
    name = datetime.now().strftime("%Y_%m_%d_%H%M") + "__ant__rnd"
    run_dir = STORAGE_DIR / name
    (run_dir / "renders").mkdir(parents=True, exist_ok=True)
    return run_dir


def run_episode(env, *, seed=None, record=False):
    """Roll out one episode with random actions. Returns (return, steps, success, frames)."""
    obs, _ = env.reset(seed=seed)
    frames = [env.render()] if record else None
    total_r, done, info = 0.0, False, {}
    while not done:
        action = env.action_space.sample()        # random [angle, magnitude] per ant
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        if record:
            frames.append(env.render())
        done = terminated or truncated
    success = info.get("object_distance", 1.0) < env.layout.reach_radius
    return total_r, info["step"], success, frames


def save_gif(frames, env, out: Path, fps: int = 30):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = env.layout.world_size
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * H / W))
    im = ax.imshow(frames[0]); ax.set_axis_off()

    def update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved GIF → {out}  ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser(description="Random-action agent for AntSwarmBarrier")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gif", dest="gif", action="store_false",
                   help="disable GIF saving (on by default)")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    env = AntSwarmEnv(seed=args.seed)

    run_dir = _make_run_dir() if args.gif else None
    if run_dir is not None:
        print(f"Run dir : {run_dir}")

    returns, steps, successes = [], [], []
    for ep in range(args.episodes):
        record = args.gif
        ret, n, ok, frames = run_episode(env, seed=args.seed + ep, record=record)
        returns.append(ret); steps.append(n); successes.append(ok)
        print(f"  ep {ep+1:3d}  return={ret:+.3f}  steps={n}  success={ok}")
        if record and frames is not None:
            out = run_dir / "renders" / f"ep_{ep+1:03d}.gif"
            print(f"  saving GIF for episode {ep+1} to {out}...")
            save_gif(frames, env, out, fps=args.fps)

    print(f"\nmean return : {np.mean(returns):+.3f} ± {np.std(returns):.3f}")
    print(f"mean steps  : {np.mean(steps):.0f}")
    print(f"success rate: {np.mean(successes)*100:.1f}%")


if __name__ == "__main__":
    main()
