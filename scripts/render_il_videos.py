"""Render evaluation videos for Imitation Learning checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gymnasium.spaces.utils import flatten
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from PIL import Image
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rl"))
from ant_swarm import AntSwarmEnv, load_config
from scripts.rl.sac_bc import SAC_BC


def save_gif(frames: list[np.ndarray], out_path: Path, fps: int = 25):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(str(out_path), save_all=True, append_images=imgs[1:], loop=0,
                 duration=max(1, int(1000 / fps)), optimize=True)
    logger.info(f"Saved GIF ({len(frames)} frames) -> {out_path}")


def render_policy_episodes(model_path: str, cfg_path: str, out_dir: str, n_episodes: int = 3):
    cfg = load_config(cfg_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading policy from {model_path}...")
    dummy_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
    model = SAC.load(model_path, env=dummy_env, device="cpu")
    dummy_env.close()

    eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=42))
    z = np.load("storage_local/datasets/successes_v1/dataset.npz")
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    for ep in range(n_episodes):
        eval_env.reset()
        base = eval_env.unwrapped
        if goals is not None:
            base.layout.goal = np.asarray(goals[ep], dtype=np.float32)
        center = np.asarray(init_poses[ep][:2], dtype=np.float32)
        angle = float(init_poses[ep][2])
        base.state.reset(center, angle)
        if base.state._bad_pose():
            continue
        base.init_center, base.init_angle = center.copy(), angle
        base.reward_model.reset(base.state)

        frames = []
        obs = flatten(base.observation_space, base.obs_model.observe(base.state)).astype(np.float32)
        first_frame = eval_env.render()
        if first_frame is not None:
            frames.append(first_frame)

        done, steps = False, 0
        while not done and steps < 300:
            pred_a, _ = model.predict(obs, deterministic=True)
            action = np.asarray(pred_a, dtype=np.float32).reshape(eval_env.action_space.shape)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
            frame = eval_env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated

        dist = info.get("object_distance", base.state.distance_to_goal())
        succ = bool(info.get("is_success", False) or dist < float(cfg.goal.reach_radius))
        logger.info(f"Episode {ep+1}: Steps={steps}, Final Dist={dist:.4f}m, Success={succ}")
        
        if frames:
            gif_name = f"episode_{ep+1:02d}_{'success' if succ else 'stuck'}.gif"
            save_gif(frames[::2], out_path / gif_name, fps=25)

    eval_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="storage_local/checkpoints/il_augmented_bc_best.zip")
    parser.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    parser.add_argument("--out", default="storage_local/renders/il_augmented_bc")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    render_policy_episodes(args.model, args.config, args.out, args.episodes)


if __name__ == "__main__":
    main()
