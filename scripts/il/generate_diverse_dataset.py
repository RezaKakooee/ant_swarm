"""Generate massive multi-zone diverse demonstration dataset and train policy on it."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces.utils import flatten
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from ant_swarm import AntSwarmEnv, load_config
from ant_swarm.geodesic import PoseGeodesicField


def generate_multizone_dataset(
    cfg,
    field: PoseGeodesicField,
    out_dataset_path: Path,
    target_trajectories: int = 20000,
):
    out_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_sparse = cfg.copy()
    cfg_sparse.env.reward_mode = "sparse"
    env = AntSwarmEnv(config=cfg_sparse, seed=123)
    reach = float(cfg.goal.reach_radius)

    zones = [
        ("Left Room", [0.06, 0.52], [0.15, 0.57], [-math.pi, math.pi], 0.35),
        ("Pre-Slit 1", [0.53, 0.74], [0.20, 0.52], [-math.pi, math.pi], 0.20),
        ("Inside Slit 1", [0.74, 0.78], [0.28, 0.44], [-math.pi, math.pi], 0.10),
        ("Middle Room", [0.78, 0.95], [0.15, 0.57], [-math.pi, math.pi], 0.15),
        ("Inside Slit 2", [0.96, 1.00], [0.28, 0.44], [-math.pi, math.pi], 0.10),
        ("Goal Room", [1.00, 1.25], [0.15, 0.57], [-math.pi, math.pi], 0.10),
    ]

    all_actions = []
    all_init_poses = []
    offsets = [0]
    successful_eps = 0

    logger.info(f"Generating {target_trajectories} multi-zone expert trajectories across 6 maze zones...")
    t0 = time.time()

    while successful_eps < target_trajectories:
        # Pick zone by weight
        zone_weights = [z[4] for z in zones]
        z_idx = np.random.choice(len(zones), p=zone_weights)
        z_name, xr, yr, tr, _ = zones[z_idx]

        # Random sample in zone
        cx = float(np.random.uniform(xr[0], xr[1]))
        cy = float(np.random.uniform(yr[0], yr[1]))
        ang = float(np.random.uniform(tr[0], tr[1]))

        # Validate reachable in field
        grid_idx = field.index(cx, cy, ang)
        if not field.reachable[grid_idx]:
            continue

        try:
            path = field.path_from([cx, cy, ang])
        except Exception:
            continue

        if len(path) < 2:
            continue

        # Execute trajectory in simulator
        env.reset()
        env.state.reset(np.asarray([cx, cy], dtype=np.float32), ang)
        if env.state._bad_pose():
            continue
        env.init_center, env.init_angle = np.array([cx, cy], dtype=np.float32), ang
        env.reward_model.reset(env.state)

        ep_actions = []
        path_idx = 0
        done, steps = False, 0

        while not done and steps < 500:
            curr_c = env.state.obj.center
            curr_a = float(env.state.obj.angle)

            # Advance along path
            while path_idx < len(path) - 1:
                target_wp = path[path_idx]
                d_pos = np.linalg.norm(curr_c - target_wp[:2])
                if d_pos < 0.03:
                    path_idx += 1
                else:
                    break

            target_wp = path[min(path_idx + 1, len(path) - 1)]
            dx = float(target_wp[0] - curr_c[0])
            dy = float(target_wp[1] - curr_c[1])
            dth = (float(target_wp[2]) - curr_a + math.pi) % (2.0 * math.pi) - math.pi

            cmd_ang = math.atan2(dy, dx) if (dx != 0 or dy != 0) else curr_a
            cmd_frc = float(np.clip(math.hypot(dx, dy) * 12.0, 0.2, 1.0))
            cmd_spn = float(np.clip(dth * 6.0, -1.0, 1.0))

            act = np.array([cmd_ang, cmd_frc, cmd_spn], dtype=np.float32)
            ep_actions.append(act)

            obs, reward, terminated, truncated, info = env.step(act.reshape(env.action_space.shape))
            steps += 1
            done = terminated or truncated

        dist = env.state.distance_to_goal()
        if dist < reach:
            successful_eps += 1
            all_init_poses.append([cx, cy, ang])
            all_actions.extend(ep_actions)
            offsets.append(len(all_actions))

            if successful_eps % 1000 == 0 or successful_eps == target_trajectories:
                rate = successful_eps / (time.time() - t0)
                logger.info(f"  Generated {successful_eps}/{target_trajectories} trajectories ({len(all_actions)} transitions, {rate:.1f} eps/sec)")

    env.close()
    np.savez_compressed(
        out_dataset_path,
        actions=np.array(all_actions, dtype=np.float32),
        init_pose=np.array(all_init_poses, dtype=np.float32),
        offsets=np.array(offsets, dtype=np.int64),
    )
    logger.info(f"Saved {successful_eps} multi-zone trajectories -> {out_dataset_path}")
    return out_dataset_path


def train_on_multizone(cfg, dataset_path: Path, out_model: Path):
    from scripts.il.train_full_dataset import collect_all_transitions, train_full, evaluate_full
    logger.info("Training Imitation Learning Actor on new Multi-Zone Dataset...")
    obs_t, act_t = collect_all_transitions(cfg, dataset_path)
    model = train_full(cfg, obs_t, act_t, out_model, epochs=25, batch_size=1024, lr=3e-4)
    evaluate_full(model, cfg, n_episodes=20)


def main():
    cfg = load_config("configs/il/il_augmented_bc.yaml")
    field = PoseGeodesicField(cfg.env.geodesic_field)
    dataset_path = Path("storage_local/datasets/diverse_multizone_v1/dataset.npz")
    out_model = Path("storage_local/checkpoints/il_diverse_multizone_best.zip")

    generate_multizone_dataset(cfg, field, dataset_path, target_trajectories=15000)
    train_on_multizone(cfg, dataset_path, out_model)


if __name__ == "__main__":
    main()
