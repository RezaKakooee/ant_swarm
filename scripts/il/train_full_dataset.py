"""Train High-Capacity Imitation Learning Actor on the full 34,777 Trajectory Dataset (4.91M transitions)."""
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


def collect_all_transitions(cfg, dataset_path: str | Path, max_episodes: int | None = None):
    z = np.load(dataset_path)
    offsets = z["offsets"]
    actions_flat = z["actions"]
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    cfg = cfg.copy()
    cfg.env.reward_mode = "sparse"
    env = AntSwarmEnv(config=cfg, seed=0)
    
    total_avail = len(offsets) - 1
    n_eps = total_avail if max_episodes is None else min(total_avail, max_episodes)
    logger.info(f"Loading full demonstration transitions from {n_eps}/{total_avail} episodes...")

    all_obs = []
    all_acts = []
    t0 = time.time()

    for i in range(n_eps):
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

        for raw_action in actions:
            obs = flatten(env.observation_space, env.obs_model.observe(env.state)).astype(np.float32)
            act = np.asarray(raw_action, dtype=np.float32).reshape(env.action_space.shape)
            all_obs.append(obs)
            all_acts.append(act.flatten())

            next_raw, reward, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                break

        if (i + 1) % 5000 == 0 or (i + 1) == n_eps:
            logger.info(f"  Loaded {i+1}/{n_eps} episodes ({len(all_obs)} transitions, {time.time()-t0:.1f}s)")

    env.close()
    obs_tensor = torch.tensor(np.array(all_obs, dtype=np.float32), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(all_acts, dtype=np.float32), dtype=torch.float32)
    logger.info(f"Complete dataset ready: {len(obs_tensor)} transitions loaded.")
    return obs_tensor, act_tensor


def train_full(
    cfg,
    obs_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    save_path: Path,
    epochs: int = 25,
    batch_size: int = 1024,
    lr: float = 3e-4,
    device: str = "auto",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training Full-Dataset Actor on {device} (batch_size={batch_size}, lr={lr:.2e})...")

    dummy_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
    sac_model = SAC(
        "MlpPolicy",
        dummy_env,
        device=device,
        learning_rate=lr,
    )
    actor = sac_model.actor.to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        actor.train()
        total_loss, n_batches = 0.0, 0
        for b_obs, b_act in loader:
            b_obs = b_obs.to(device)
            b_act = b_act.to(device)

            optimizer.zero_grad()
            mean_actions, _, _ = actor.get_action_dist_params(b_obs)
            squashed = torch.tanh(mean_actions)
            pred_ang = squashed[:, 0] * math.pi
            pred_frc = (squashed[:, 1] + 1.0) * 0.5
            pred_spn = squashed[:, 2]
            pred_act = torch.stack([pred_ang, pred_frc, pred_spn], dim=-1)

            loss = loss_fn(pred_act, b_act)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        logger.info(f"Epoch {ep:2d}/{epochs:2d} - Full Dataset MSE Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            sac_model.save(save_path)

    logger.info(f"Saved Best Full-Dataset Model (MSE={best_loss:.6f}) -> {save_path}")
    dummy_env.close()
    return sac_model


def evaluate_full(model: SAC, cfg, n_episodes: int = 20):
    logger.info(f"Evaluating Full-Dataset Policy on {n_episodes} closed-loop episodes...")
    z = np.load("storage_local/datasets/successes_v1/dataset.npz")
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=42))
    reach = float(eval_env.unwrapped.layout.reach_radius)
    distances, successes = [], []

    _, first = np.unique(init_poses, axis=0, return_index=True)
    eval_ids = np.sort(np.random.default_rng(0).choice(
        first, size=min(n_episodes, len(first)), replace=False))
    for ep in eval_ids:
        base = eval_env.unwrapped
        options = {"init_pose": init_poses[ep]}
        if goals is not None:
            options["goal"] = goals[ep]
        eval_env.reset(seed=42 + int(ep), options=options)

        obs = flatten(base.observation_space, base.obs_model.observe(base.state)).astype(np.float32)
        done, steps = False, 0
        while not done and steps < 500:
            pred_a, _ = model.predict(obs, deterministic=True)
            action = np.asarray(pred_a, dtype=np.float32).reshape(eval_env.action_space.shape)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
            done = terminated or truncated

        dist = info.get("object_distance", base.state.distance_to_goal())
        succ = bool(info.get("is_success", False) or dist < reach)
        distances.append(dist)
        successes.append(succ)
        logger.info(f"  Ep {ep+1:02d}: Final Dist={dist:.4f}m, Success={succ}")

    eval_env.close()
    succ_rate = np.mean(successes) * 100.0
    mean_dist = np.mean(distances)
    logger.info(f"[Full-Dataset Evaluation] Success Rate={succ_rate:.1f}% ({np.sum(successes)}/{len(successes)}), Mean Dist={mean_dist:.4f}m")
    return succ_rate, mean_dist


def main():
    cfg = load_config("configs/il/il_augmented_bc.yaml")
    save_path = Path("storage_local/checkpoints/il_full_dataset_best.zip")
    obs_t, act_t = collect_all_transitions(cfg, "storage_local/datasets/successes_v1/dataset.npz")
    model = train_full(cfg, obs_t, act_t, save_path, epochs=25, batch_size=1024, lr=3e-4)
    evaluate_full(model, cfg, n_episodes=20)


if __name__ == "__main__":
    main()
