"""Train a Behavior Cloning (BC) policy on demonstration dataset.

Fits a neural network actor on state-action pairs from
``storage_local/datasets/successes_v1/dataset.npz``, and saves an SB3 SAC-compatible
checkpoint so that `run.transfer_actor_from` or direct evaluation can use it.

Usage:
    python scripts/il/train_bc.py [--epochs 20] [--batch-size 256] [--lr 1e-3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces.utils import flatten
from loguru import logger
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from ant_swarm import AntSwarmEnv, load_config


def collect_transitions(cfg, dataset_path: str | Path, max_episodes: int = 300):
    z = np.load(dataset_path)
    offsets = z["offsets"]
    actions_flat = z["actions"]
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    # Use sparse reward mode during collection to skip expensive geodesic field queries
    cfg = cfg.copy()
    cfg.env.reward_mode = "sparse"
    env = AntSwarmEnv(config=cfg, seed=0)
    n_eps = min(len(offsets) - 1, max_episodes)
    logger.info(f"Collecting transitions from {n_eps} demonstration episodes...")

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

        obs = flatten(env.observation_space, env.obs_model.observe(env.state)).astype(np.float32)
        for raw_action in actions:
            action = np.asarray(raw_action, dtype=np.float32).reshape(env.action_space.shape)
            all_obs.append(obs)
            all_acts.append(raw_action)
            next_raw, reward, terminated, truncated, info = env.step(action)
            obs = flatten(env.observation_space, next_raw).astype(np.float32)
            if terminated or truncated:
                break
        if (i + 1) % 100 == 0 or (i + 1) == n_eps:
            logger.info(f"  Processed {i+1}/{n_eps} episodes ({len(all_obs)} transitions, {time.time()-t0:.1f}s)")

    env.close()
    obs_tensor = torch.tensor(np.array(all_obs, dtype=np.float32), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(all_acts, dtype=np.float32), dtype=torch.float32)
    logger.info(f"Collected {len(obs_tensor)} transitions in {time.time()-t0:.1f}s")
    return obs_tensor, act_tensor


def train_bc(
    cfg,
    obs_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    save_path: Path,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "auto",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training BC model on device: {device}")

    # Build dummy SAC model with FlattenObservation to match SAC policy observation space
    dummy_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
    sac_model = SAC(
        "MlpPolicy",
        dummy_env,
        device=device,
        learning_rate=lr,
    )
    actor = sac_model.actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    actor.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for b_obs, b_act in loader:
            b_obs = b_obs.to(device)
            b_act = b_act.to(device)

            optimizer.zero_grad()
            # SB3 SAC actor predict / mean action
            mean_actions, _, _ = actor.get_action_dist_params(b_obs)
            loss = loss_fn(mean_actions, b_act)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        if ep % 5 == 0 or ep == epochs:
            logger.info(f"Epoch {ep:2d}/{epochs:2d} - BC MSE Loss: {avg_loss:.6f}")

    # Save as standard SB3 SAC zip file
    sac_model.save(save_path)
    logger.info(f"Saved BC pre-trained SAC model -> {save_path}")
    dummy_env.close()
    return sac_model


def evaluate_bc(model, cfg, n_episodes: int = 20):
    logger.info(f"Evaluating BC policy on {n_episodes} random test episodes...")
    from gymnasium.wrappers import FlattenObservation
    env = FlattenObservation(AntSwarmEnv(config=cfg, seed=999))
    reach = float(cfg.goal.reach_radius)
    returns, lengths, successes = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r, done = 0.0, False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            steps += 1
            done = terminated or truncated

        dist = info.get("object_distance", 999.0)
        succ = bool(info.get("is_success", False) or dist < reach)
        returns.append(total_r)
        lengths.append(steps)
        successes.append(succ)

    env.close()
    succ_rate = np.mean(successes) * 100.0
    logger.info(f"BC Evaluation ({n_episodes} eps): Success Rate={succ_rate:.1f}%, Mean Steps={np.mean(lengths):.1f}")
    return succ_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/gen_e_scratch_randall.yaml")
    parser.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    parser.add_argument("--out", default="storage_local/checkpoints/bc_actor_model.zip")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    obs, acts = collect_transitions(cfg, args.dataset, max_episodes=args.episodes)
    model = train_bc(
        cfg,
        obs,
        acts,
        save_path=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    evaluate_bc(model, cfg, n_episodes=20)


if __name__ == "__main__":
    main()
