#!/usr/bin/env python3
"""Diagnostic script to investigate why Option A and Option B failed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces.utils import flatten
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ant_swarm import AntSwarmEnv, load_config


def main():
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC INVESTIGATION: DEMONSTRATIONS & POLICIES")
    logger.info("=" * 60)

    dataset_path = PROJECT_ROOT / "storage_local/datasets/successes_v1/dataset.npz"
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    z = np.load(dataset_path)
    offsets = z["offsets"]
    actions_flat = z["actions"]
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None
    n_trajs = len(offsets) - 1

    logger.info(f"Dataset loaded: {n_trajs} trajectories, {len(actions_flat)} total steps")
    logger.info(f"Action ranges: min={actions_flat.min(axis=0)}, max={actions_flat.max(axis=0)}")
    logger.info(f"Init poses: x in [{init_poses[:,0].min():.3f}, {init_poses[:,0].max():.3f}], y in [{init_poses[:,1].min():.3f}, {init_poses[:,1].max():.3f}], theta in [{init_poses[:,2].min():.3f}, {init_poses[:,2].max():.3f}]")
    if goals is not None:
        logger.info(f"Goals: x in [{goals[:,0].min():.3f}, {goals[:,0].max():.3f}], y in [{goals[:,1].min():.3f}, {goals[:,1].max():.3f}]")

    # 1. Open-loop replay test of the demonstration trajectories
    cfg = load_config("configs/rl/gen_m_demo_seeded.yaml")
    env = AntSwarmEnv(config=cfg, seed=0)
    reach = float(cfg.goal.reach_radius)
    n_test = min(100, n_trajs)
    logger.info(f"\n--- 1. Testing Open-Loop Replay of {n_test} Demonstrations in AntSwarmEnv ---")

    replay_successes = []
    final_dists = []
    lengths = []

    for i in range(n_test):
        actions = actions_flat[offsets[i]:offsets[i+1]]
        init_pose = init_poses[i]
        goal = goals[i] if goals is not None else None

        env.reset()
        if goal is not None:
            env.layout.goal = np.asarray(goal, dtype=np.float32)
        center = np.asarray(init_pose[:2], dtype=np.float32)
        angle = float(init_pose[2])
        env.state.reset(center, angle)
        if env.state._bad_pose():
            continue
        env.init_center, env.init_angle = center.copy(), angle
        env.reward_model.reset(env.state)

        for act in actions:
            action = np.asarray(act, dtype=np.float32).reshape(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        dist = info.get("object_distance", env.state.distance_to_goal())
        succ = bool(info.get("is_success", False) or dist < reach)
        replay_successes.append(succ)
        final_dists.append(dist)
        lengths.append(len(actions))

    env.close()
    logger.info(f"Open-Loop Replay Success Rate: {np.mean(replay_successes)*100:.1f}% ({sum(replay_successes)}/{len(replay_successes)})")
    logger.info(f"Mean Final Dist: {np.mean(final_dists):.4f} m (min={np.min(final_dists):.4f}, max={np.max(final_dists):.4f})")
    logger.info(f"Mean Demo Length: {np.mean(lengths):.1f} steps")

    # 2. Test Pre-Trained BC Policy on Demo Init Poses vs Random Init Poses
    bc_model_path = PROJECT_ROOT / "storage_local/checkpoints/bc_actor_model.zip"
    if bc_model_path.exists():
        logger.info(f"\n--- 2. Evaluating Pre-trained BC Actor Model ({bc_model_path.name}) ---")
        eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=42))
        bc_model = SAC.load(
            str(bc_model_path),
            env=eval_env,
            custom_objects={"observation_space": eval_env.observation_space, "action_space": eval_env.action_space},
        )

        # 2a. Evaluated on Demonstration Initial Poses
        demo_eval_succ = []
        demo_eval_dist = []
        for i in range(min(50, n_trajs)):
            eval_env.reset()
            base_env = eval_env.unwrapped
            if goals is not None:
                base_env.layout.goal = np.asarray(goals[i], dtype=np.float32)
            center = np.asarray(init_poses[i][:2], dtype=np.float32)
            angle = float(init_poses[i][2])
            base_env.state.reset(center, angle)
            if base_env.state._bad_pose():
                continue
            base_env.init_center, base_env.init_angle = center.copy(), angle
            base_env.reward_model.reset(base_env.state)

            obs = flatten(base_env.observation_space, base_env.obs_model.observe(base_env.state)).astype(np.float32)
            done = False
            steps = 0
            while not done and steps < 500:
                action, _ = bc_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                steps += 1
                done = terminated or truncated

            dist = info.get("object_distance", base_env.state.distance_to_goal())
            succ = bool(info.get("is_success", False) or dist < reach)
            demo_eval_succ.append(succ)
            demo_eval_dist.append(dist)

        logger.info(f"BC Policy on DEMO Initial Poses: Success Rate={np.mean(demo_eval_succ)*100:.1f}%, Mean Final Dist={np.mean(demo_eval_dist):.4f} m")

        # 2b. Evaluated on RANDOM Full-Maze Initial Poses
        rand_eval_succ = []
        rand_eval_dist = []
        for _ in range(50):
            obs, _ = eval_env.reset()
            done = False
            steps = 0
            while not done and steps < 500:
                action, _ = bc_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                steps += 1
                done = terminated or truncated

            dist = info.get("object_distance", 999.0)
            succ = bool(info.get("is_success", False) or dist < reach)
            rand_eval_succ.append(succ)
            rand_eval_dist.append(dist)

        logger.info(f"BC Policy on RANDOM Full-Maze Poses: Success Rate={np.mean(rand_eval_succ)*100:.1f}%, Mean Final Dist={np.mean(rand_eval_dist):.4f} m")
        eval_env.close()

    # 3. Evaluate Final SAC Checkpoints
    for opt_name, ckpt_rel in [
        ("Option A (Seeded SAC Final)", "storage_local/ant__20260825_2306__26559__train_sac__gen_m_demo_seeded/checkpoints/sac_final.zip"),
        ("Option B (BC+SAC Fine-Tuned Final)", "storage_local/ant__20260825_2312__26775__train_sac__gen_n_bc_transfer/checkpoints/sac_final.zip"),
    ]:
        ckpt_path = PROJECT_ROOT / ckpt_rel
        if not ckpt_path.exists():
            continue
        logger.info(f"\n--- 3. Evaluating {opt_name} ---")
        eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=42))
        model = SAC.load(
            str(ckpt_path),
            env=eval_env,
            custom_objects={"observation_space": eval_env.observation_space, "action_space": eval_env.action_space},
        )
        succs, dists = [], []
        for _ in range(50):
            obs, _ = eval_env.reset()
            done, steps = False, 0
            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                steps += 1
                done = terminated or truncated
            dist = info.get("object_distance", 999.0)
            succ = bool(info.get("is_success", False) or dist < reach)
            succs.append(succ)
            dists.append(dist)
        logger.info(f"{opt_name} (50 random episodes): Success Rate={np.mean(succs)*100:.1f}%, Mean Final Dist={np.mean(dists):.4f} m")
        eval_env.close()


if __name__ == "__main__":
    main()
