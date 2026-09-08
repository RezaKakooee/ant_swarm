"""High-Performance Imitation Learning (BC) on Full Demonstration Dataset.

Trains an actor network on up to 10,000 demonstration trajectories with exact
end-to-end action space matching (actor(obs, deterministic=True)), eliminating
tanh distortion and compounding rotational errors.
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
from gymnasium.wrappers import FlattenObservation
from loguru import logger
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from ant_swarm import AntSwarmEnv, load_config


def collect_transitions(cfg, dataset_path: str | Path, max_episodes: int = 5000):
    z = np.load(dataset_path)
    offsets = z["offsets"]
    actions_flat = z["actions"]
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

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
            all_acts.append(action.flatten())
            next_raw, reward, terminated, truncated, info = env.step(action)
            obs = flatten(env.observation_space, next_raw).astype(np.float32)
            if terminated or truncated:
                break
        if (i + 1) % 500 == 0 or (i + 1) == n_eps:
            logger.info(f"  Processed {i+1}/{n_eps} episodes ({len(all_obs)} transitions, {time.time()-t0:.1f}s)")

    env.close()
    obs_tensor = torch.tensor(np.array(all_obs, dtype=np.float32), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(all_acts, dtype=np.float32), dtype=torch.float32)
    logger.info(f"Collected {len(obs_tensor)} transitions in {time.time()-t0:.1f}s")
    return obs_tensor, act_tensor


def train_il(
    cfg,
    obs_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    save_path: Path,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str = "auto",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training Imitation Learning model on device: {device}")

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
        total_loss = 0.0
        n_batches = 0
        for b_obs, b_act in loader:
            b_obs = b_obs.to(device)
            b_act = b_act.to(device)

            optimizer.zero_grad()
            # Exact end-to-end forward matching environment action execution
            pred_actions = actor(b_obs, deterministic=True)
            loss = loss_fn(pred_actions, b_act)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        if ep % 5 == 0 or ep == epochs:
            logger.info(f"Epoch {ep:2d}/{epochs:2d} - IL MSE Loss: {avg_loss:.6f} (lr={scheduler.get_last_lr()[0]:.2e})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            sac_model.save(save_path)

    logger.info(f"Saved best IL model (MSE={best_loss:.6f}) -> {save_path}")
    dummy_env.close()
    return sac_model


def evaluate_il(model, cfg, n_episodes: int = 20):
    logger.info(f"Evaluating IL policy on {n_episodes} closed-loop test episodes...")
    z = np.load("storage_local/datasets/successes_v1/dataset.npz")
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=42))
    reach = float(cfg.goal.reach_radius)
    returns, lengths, successes = [], [], []

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

        obs = flatten(base.observation_space, base.obs_model.observe(base.state)).astype(np.float32)
        done = False
        steps = 0
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
            done = terminated or truncated

        dist = info.get("object_distance", base.state.distance_to_goal())
        succ = bool(info.get("is_success", False) or dist < reach)
        lengths.append(steps)
        successes.append(succ)
        print(f"Episode {ep+1:02d}: Steps={steps:03d}, Final Dist={dist:.4f}m, Success={succ}")

    eval_env.close()
    succ_rate = np.mean(successes) * 100.0
    logger.info(f"IL Closed-Loop Evaluation ({n_episodes} eps): Success Rate={succ_rate:.1f}%, Mean Steps={np.mean(lengths):.1f}")
    return succ_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/gen_m_demo_seeded.yaml")
    parser.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    parser.add_argument("--out", default="storage_local/checkpoints/il_actor_best.zip")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_path = Path(args.out)
    obs_tensor, act_tensor = collect_transitions(cfg, args.dataset, max_episodes=args.episodes)
    model = train_il(
        cfg,
        obs_tensor,
        act_tensor,
        save_path=save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    evaluate_il(model, cfg, n_episodes=20)


if __name__ == "__main__":
    main()
