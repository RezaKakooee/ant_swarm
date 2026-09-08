"""Frame-Stacked Imitation Learning (History Window K=4) on Full Demonstration Dataset.

Stacks K=4 consecutive observation frames [s_{t-3}, s_{t-2}, s_{t-1}, s_t] to provide
the actor with temporal momentum and angular acceleration awareness, eliminating
angular drift when entering narrow slits.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from ant_swarm import AntSwarmEnv, load_config


class FrameStackWrapper:
    """Standard K-frame history stack wrapper."""

    def __init__(self, env: AntSwarmEnv, k: int = 4):
        self.env = env
        self.k = k
        self.frames = deque(maxlen=k)
        self.base_dim = env.obs_model.obs_dim
        self.obs_dim = self.base_dim * k

    def reset(self, center=None, angle=None):
        if center is not None and angle is not None:
            self.env.state.reset(center, angle)
        obs_raw = self.env.obs_model.observe(self.env.state)
        obs = flatten(self.env.observation_space, obs_raw).astype(np.float32)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        next_raw, reward, terminated, truncated, info = self.env.step(action)
        obs = flatten(self.env.observation_space, next_raw).astype(np.float32)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


def collect_framestack_transitions(cfg, dataset_path: str | Path, k: int = 4, max_episodes: int | None = None):
    z = np.load(dataset_path)
    offsets = z["offsets"]
    actions_flat = z["actions"]
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    cfg = cfg.copy()
    cfg.env.reward_mode = "sparse"
    base_env = AntSwarmEnv(config=cfg, seed=0)
    stack_env = FrameStackWrapper(base_env, k=k)

    total_avail = len(offsets) - 1
    n_eps = total_avail if max_episodes is None else min(total_avail, max_episodes)
    logger.info(f"Collecting K={k} Frame-Stacked transitions from {n_eps}/{total_avail} demonstration episodes...")

    all_obs = []
    all_acts = []
    t0 = time.time()

    for i in range(n_eps):
        actions = actions_flat[offsets[i]:offsets[i+1]]
        init_pose = init_poses[i]
        goal = goals[i] if goals is not None else None

        base_env.reset()
        if goal is not None:
            base_env.layout.goal = np.asarray(goal, dtype=np.float32)
        center = np.asarray(init_pose[:2], dtype=np.float32)
        angle = float(init_pose[2])
        base_env.state.reset(center, angle)
        if base_env.state._bad_pose():
            continue
        base_env.init_center, base_env.init_angle = center.copy(), angle
        base_env.reward_model.reset(base_env.state)

        stacked_obs = stack_env.reset(center, angle)

        for raw_action in actions:
            act = np.asarray(raw_action, dtype=np.float32).reshape(base_env.action_space.shape)
            all_obs.append(stacked_obs)
            all_acts.append(act.flatten())

            stacked_obs, reward, terminated, truncated, info = stack_env.step(act)
            if terminated or truncated:
                break

        if (i + 1) % 5000 == 0 or (i + 1) == n_eps:
            logger.info(f"  Processed {i+1}/{n_eps} episodes ({len(all_obs)} transitions, {time.time()-t0:.1f}s)")

    base_env.close()
    obs_tensor = torch.tensor(np.array(all_obs, dtype=np.float32), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(all_acts, dtype=np.float32), dtype=torch.float32)
    logger.info(f"Frame-Stacked dataset ready: {len(obs_tensor)} transitions (Input Dim={obs_tensor.shape[1]}).")
    return obs_tensor, act_tensor


class FrameStackedSACActor(nn.Module):
    """Deep MLP Actor for Frame-Stacked observations."""

    def __init__(self, obs_dim: int, act_dim: int = 3, net_arch: list[int] = [512, 512, 256]):
        super().__init__()
        layers = []
        in_d = obs_dim
        for h in net_arch:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_d = h
        self.latent = nn.Sequential(*layers)
        self.mu = nn.Linear(in_d, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.latent(obs)
        raw_act = self.mu(feat)
        squashed = torch.tanh(raw_act)
        # Rescale: angle [-pi, pi], force [0, 1], spin [-1, 1]
        pred_ang = squashed[:, 0] * math.pi
        pred_frc = (squashed[:, 1] + 1.0) * 0.5
        pred_spn = squashed[:, 2]
        return torch.stack([pred_ang, pred_frc, pred_spn], dim=-1)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            pred = self.forward(t_obs).squeeze(0).cpu().numpy()
        return pred


def train_framestack_actor(
    obs_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    save_path: Path,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 3e-4,
    device: str = "auto",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training Frame-Stacked Actor on {device} (Input Dim={obs_tensor.shape[1]})...")

    actor = FrameStackedSACActor(obs_dim=obs_tensor.shape[1], act_dim=3).to(device)
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
            pred_act = actor(b_obs)
            loss = loss_fn(pred_act, b_act)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        logger.info(f"Epoch {ep:2d}/{epochs:2d} - FrameStack MSE Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(actor.state_dict(), save_path)

    logger.info(f"Saved Best Frame-Stacked Actor (MSE={best_loss:.6f}) -> {save_path}")
    return actor


def evaluate_framestack_policy(actor: FrameStackedSACActor, cfg, k: int = 4, n_episodes: int = 20):
    logger.info(f"Evaluating Frame-Stacked Policy (K={k}) on {n_episodes} closed-loop episodes...")
    z = np.load("storage_local/datasets/successes_v1/dataset.npz")
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    base_env = AntSwarmEnv(config=cfg, seed=42)
    stack_env = FrameStackWrapper(base_env, k=k)
    reach = float(cfg.goal.reach_radius)
    distances, successes = [], []

    for ep in range(n_episodes):
        base_env.reset()
        if goals is not None:
            base_env.layout.goal = np.asarray(goals[ep], dtype=np.float32)
        center = np.asarray(init_poses[ep][:2], dtype=np.float32)
        angle = float(init_poses[ep][2])
        base_env.state.reset(center, angle)
        if base_env.state._bad_pose():
            continue
        base_env.init_center, base_env.init_angle = center.copy(), angle
        base_env.reward_model.reset(base_env.state)

        stacked_obs = stack_env.reset(center, angle)
        done, steps = False, 0

        while not done and steps < 500:
            pred_a = actor.predict(stacked_obs)
            action = np.asarray(pred_a, dtype=np.float32).reshape(base_env.action_space.shape)
            stacked_obs, reward, terminated, truncated, info = stack_env.step(action)
            steps += 1
            done = terminated or truncated

        dist = info.get("object_distance", base_env.state.distance_to_goal())
        succ = bool(info.get("is_success", False) or dist < reach)
        distances.append(dist)
        successes.append(succ)
        logger.info(f"  Ep {ep+1:02d}: Steps={steps:03d}, Final Dist={dist:.4f}m, Success={succ}")

    base_env.close()
    succ_rate = np.mean(successes) * 100.0
    mean_dist = np.mean(distances)
    logger.info(f"[FrameStack K={k} Evaluation] Success Rate={succ_rate:.1f}% ({np.sum(successes)}/{n_episodes}), Mean Dist={mean_dist:.4f}m")
    return succ_rate, mean_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    parser.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default="storage_local/checkpoints/il_framestack_k4_best.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_path = Path(args.out)

    obs_tensor, act_tensor = collect_framestack_transitions(cfg, args.dataset, k=args.k)
    actor = train_framestack_actor(
        obs_tensor,
        act_tensor,
        save_path=save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    evaluate_framestack_policy(actor, cfg, k=args.k, n_episodes=20)


if __name__ == "__main__":
    main()
