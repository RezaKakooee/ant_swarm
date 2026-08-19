"""Standalone evaluation runner for Ant-Swarm RL policies.

Evaluates trained SAC/PPO policies on the full-difficulty task, logs per-episode
metrics, and records MP4 & GIF video animations saved directly in the experiment
directory.

Usage:
    python scripts/rl/evaluate.py <exp_dir_or_checkpoint> [--episodes N] [--fps N] [--stochastic]

Examples:
    python scripts/rl/evaluate.py storage_local/ant__20260814_1230__local-1427853__train_sac__pnas_dyn_geo_v2
    python scripts/rl/evaluate.py storage_local/<run>/checkpoints/best/best_model.zip --episodes 5
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from stable_baselines3 import PPO, SAC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ant_swarm import AntSwarmEnv, load_config, setup_logging  # noqa: E402
from scripts.rl.train_utils import build_curriculum, pin_standalone_eval_hard  # noqa: E402


def _resolve_target(target_str: str) -> tuple[Path, Path, str]:
    """Find checkpoint file, experiment root directory, and algorithm ('sac' or 'ppo')."""
    target = Path(target_str).expanduser()
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()

    if target.is_dir():
        exp_dir = target
        candidates = [
            (target / "checkpoints" / "best" / "best_model.zip", None),
            (target / "checkpoints" / "sac_final.zip", "sac"),
            (target / "checkpoints" / "ppo_final.zip", "ppo"),
            (target / "best" / "best_model.zip", None),
            (target / "best_model.zip", None),
            (target / "sac_final.zip", "sac"),
            (target / "ppo_final.zip", "ppo"),
        ]
        for c, algo in candidates:
            if c.is_file():
                if algo is None:
                    algo = "sac" if "sac" in exp_dir.name.lower() else "ppo"
                return c, exp_dir, algo
        
        # Check any zip in checkpoints/
        zips = sorted((target / "checkpoints").glob("*.zip"))
        if zips:
            algo = "sac" if "sac" in exp_dir.name.lower() else "ppo"
            return zips[-1], exp_dir, algo
        raise FileNotFoundError(f"No checkpoint .zip found inside directory: {target}")

    # Target is a file
    model_path = target
    exp_dir = model_path.parent
    if exp_dir.name in ("best", "eval_logs"):
        exp_dir = exp_dir.parent
    if exp_dir.name == "checkpoints":
        exp_dir = exp_dir.parent

    algo = "sac" if ("sac" in model_path.name.lower() or "sac" in exp_dir.name.lower()) else "ppo"
    return model_path, exp_dir, algo


def _save_gif(frames: list[np.ndarray], out_path: Path, fps: int = 30):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(str(out_path), save_all=True, append_images=imgs[1:], loop=0,
                 duration=max(1, int(1000 / fps)), optimize=True)


def _save_mp4(frames: list[np.ndarray], out_path: Path, fps: int = 30):
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


def _save_grid_gif(all_episode_frames: list[list[np.ndarray]], out_path: Path, fps: int = 30,
                   tile_w: int = 320, frame_cap: int = 200):
    """Tile all evaluated episodes into a synchronized grid GIF."""
    if not all_episode_frames:
        return
    import cv2
    n = len(all_episode_frames)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    resized_clips = []
    for ep_frames in all_episode_frames:
        stride = max(1, len(ep_frames) // frame_cap)
        sub = ep_frames[::stride]
        h, w = sub[0].shape[:2]
        tile_h = int(tile_w * h / w)
        scaled = [cv2.resize(f, (tile_w, tile_h), interpolation=cv2.INTER_AREA) for f in sub]
        resized_clips.append(scaled)

    tile_h, tile_w = resized_clips[0][0].shape[:2]
    max_len = max(len(c) for c in resized_clips)

    tiled_frames = []
    for t in range(max_len):
        canvas = np.full((rows * tile_h, cols * tile_w, 3), 240, dtype=np.uint8)
        for k, clip in enumerate(resized_clips):
            frame = clip[min(t, len(clip) - 1)]
            r, c = divmod(k, cols)
            canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = frame
        tiled_frames.append(canvas)

    _save_gif(tiled_frames, out_path, fps=fps)
    _save_mp4(tiled_frames, out_path.with_suffix(".mp4"), fps=fps)


def run_evaluation(target: str, episodes: int = 5, fps: int = 30,
                   deterministic: bool = True, out_dir: str | Path | None = None):
    setup_logging()
    model_file, exp_dir, algo = _resolve_target(target)
    
    logger.info("=" * 60)
    logger.info("Ant-Swarm Policy Evaluation")
    logger.info(f"Target Checkpoint : {model_file}")
    logger.info(f"Experiment Dir    : {exp_dir}")
    logger.info(f"Algorithm         : {algo.upper()}")
    logger.info(f"Episodes          : {episodes}")
    logger.info(f"Deterministic     : {deterministic}")
    logger.info("=" * 60)

    # Load resolved config from experiment snapshot if available
    snap_cfg = exp_dir / "code" / "config.yaml"
    if snap_cfg.is_file():
        logger.info(f"Loading snapshot config: {snap_cfg}")
        cfg = load_config(snap_cfg)
    else:
        logger.info("Using default config")
        cfg = load_config()

    reach = float(cfg.goal.reach_radius)
    raw_env = AntSwarmEnv(config=cfg, seed=42)
    cur = getattr(cfg, "curriculum", None)
    if cur is not None and getattr(cur, "enabled", False):
        curriculum = build_curriculum(cur, reach, cfg=cfg)
        pin_standalone_eval_hard(raw_env, cur, curriculum)

    env = FlattenObservation(raw_env)

    if algo == "sac":
        model = SAC.load(str(model_file), env=env)
    else:
        model = PPO.load(str(model_file), env=env)

    eval_out_dir = Path(out_dir) if out_dir else (exp_dir / "eval")
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    returns, lengths, successes, all_frames = [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep * 100)
        frames = [raw_env.render()]
        total_r, done = 0.0, False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(raw_env.render())
            total_r += reward
            done = terminated or truncated

        step_count = info.get("step", len(frames) - 1)
        final_dist = info.get("object_distance", float("nan"))
        succ = bool(final_dist < reach)

        returns.append(total_r)
        lengths.append(step_count)
        successes.append(succ)
        all_frames.append(frames)

        ep_gif = eval_out_dir / f"eval_ep{ep + 1:02d}_len{step_count}_ret{total_r:.2f}.gif"
        ep_mp4 = eval_out_dir / f"eval_ep{ep + 1:02d}_len{step_count}_ret{total_r:.2f}.mp4"
        _save_gif(frames, ep_gif, fps=fps)
        _save_mp4(frames, ep_mp4, fps=fps)

        status_tag = "SUCCESS" if succ else "FAILED"
        logger.info(
            f"Episode {ep + 1:2d}/{episodes:2d} | Return: {total_r:6.2f} | "
            f"Steps: {step_count:4d} | Final Dist: {final_dist:6.3f} | [{status_tag}]"
        )
        logger.info(f"  → Video: {ep_gif.relative_to(PROJECT_ROOT) if ep_gif.is_relative_to(PROJECT_ROOT) else ep_gif}")

    # Generate multi-episode grid video if > 1 episode
    if len(all_frames) > 1:
        grid_gif = eval_out_dir / f"eval_all_{episodes}episodes_grid.gif"
        _save_grid_gif(all_frames, grid_gif, fps=fps)
        logger.info(f"  → Summary Grid: {grid_gif.relative_to(PROJECT_ROOT) if grid_gif.is_relative_to(PROJECT_ROOT) else grid_gif}")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print(f"  Checkpoints evaluated : {model_file.name}")
    print(f"  Episodes count        : {episodes}")
    print(f"  Success rate          : {np.mean(successes) * 100:.1f}% ({sum(successes)}/{episodes})")
    print(f"  Mean episode return   : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Mean episode length   : {np.mean(lengths):.1f} steps")
    print(f"  All videos saved to   : {eval_out_dir}")
    print("=" * 60 + "\n")

    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "eval_dir": str(eval_out_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Ant-Swarm policy and record videos.")
    parser.add_argument("target", type=str, help="Experiment directory or checkpoint .zip path")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30)")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy sampling instead of greedy deterministic")
    parser.add_argument("--out-dir", "-o", type=str, default=None, help="Output directory for videos (default: <exp>/eval)")

    args = parser.parse_args()
    run_evaluation(
        target=args.target,
        episodes=args.episodes,
        fps=args.fps,
        deterministic=not args.stochastic,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
