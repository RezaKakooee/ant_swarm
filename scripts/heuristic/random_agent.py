"""Random agent for the AntSwarmBarrier env — samples actions uniformly.

A baseline / sanity-check policy: every step each ant gets a random
``[angle, magnitude]`` from the action space.

Saves one GIF per episode by default into a timestamped run dir under
``storage_local/<date>__ant__rnd/renders/``.

Usage (hydra-style overrides; config: configs/heuristic/random_agent.yaml):
    python scripts/heuristic/random_agent.py                      # 5 episodes, one GIF each
    python scripts/heuristic/random_agent.py heuristic.episodes=20
    python scripts/heuristic/random_agent.py heuristic.gif=false  # stats only
    python scripts/heuristic/random_agent.py heuristic.episodes=3 heuristic.seed=3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ant_swarm import (AntSwarmEnv, build_run_id, load_config_cli,  # noqa: E402
                       save_code, setup_logging)
from loguru import logger  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = PROJECT_ROOT / "storage_local"


def _make_run_dir(n_ants: int) -> Path:
    """Run dir under storage_local, named by the shared run id (ant_swarm/run_id.py)."""
    run_dir = STORAGE_DIR / build_run_id("rnd", n_ants)
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
    logger.info(f"saved GIF → {out}  ({len(frames)} frames)")


def main():
    setup_logging()
    cfg = load_config_cli(config_dir=PROJECT_ROOT / "configs" / "heuristic",
                          default_name="random_agent")
    h = cfg.get("heuristic", {})
    episodes = int(h.get("episodes", 5))
    seed = int(h.get("seed", 0))
    gif = bool(h.get("gif", True))
    fps = int(h.get("fps", 30))

    env = AntSwarmEnv(config=cfg, seed=seed)

    run_dir = _make_run_dir(env.n_ants) if gif else None
    if run_dir is not None:
        setup_logging(run_dir)
        save_code(run_dir, __file__, cfg=cfg)   # snapshot code + resolved config
        logger.info(f"Run dir : {run_dir}")

    returns, steps, successes = [], [], []
    for ep in range(episodes):
        ret, n, ok, frames = run_episode(env, seed=seed + ep, record=gif)
        returns.append(ret); steps.append(n); successes.append(ok)
        logger.info(f"ep {ep+1:3d}  return={ret:+.3f}  steps={n}  success={ok}")
        if gif and frames is not None:
            out = run_dir / "renders" / f"ep_{ep+1:03d}.gif"
            save_gif(frames, env, out, fps=fps)

    logger.info(f"mean return : {np.mean(returns):+.3f} ± {np.std(returns):.3f}")
    logger.info(f"mean steps  : {np.mean(steps):.0f}")
    logger.info(f"success rate: {np.mean(successes)*100:.1f}%")


if __name__ == "__main__":
    main()
