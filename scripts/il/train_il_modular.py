"""Modular Imitation Learning Framework for Ant-Swarm (No Curriculum).

Supports:
  1. Method 1: Perturbation-Augmented Behavior Cloning (Augmented BC)
  2. Method 2: DAgger (Interactive Dataset Aggregation)
"""
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


def get_expert_action(field: PoseGeodesicField, env: AntSwarmEnv) -> np.ndarray:
    """Query the geodesic distance field to get the optimal descent action from current pose."""
    obj = env.state.obj
    x, y, theta = float(obj.center[0]), float(obj.center[1]), float(obj.angle)
    
    current_idx = field.index(x, y, theta)
    if not field.reachable[current_idx]:
        # If clipped into a wall, try nearest reachable neighbor
        reachable_neighbors = [q for q in field._neighbors(current_idx) if field.reachable[q]]
        if reachable_neighbors:
            current_idx = min(reachable_neighbors, key=lambda q: int(field.steps[q]))
        else:
            # Fallback: push toward goal
            gx, gy = env.layout.goal
            ang = math.atan2(gy - y, gx - x)
            return np.array([ang, 1.0, 0.0], dtype=np.float32)

    remaining = int(field.steps[current_idx])
    descending = [q for q in field._neighbors(current_idx) if field.reachable[q] and int(field.steps[q]) < remaining]
    if not descending:
        descending = [q for q in field._neighbors(current_idx) if field.reachable[q]]
    
    if descending:
        best_idx = min(descending, key=lambda q: int(field.steps[q]))
        target_pose = field.pose(best_idx)
        dx = float(target_pose[0]) - x
        dy = float(target_pose[1]) - y
        dth = (float(target_pose[2]) - theta + math.pi) % (2.0 * math.pi) - math.pi
        
        angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else theta
        force = 1.0
        spin = float(np.clip(dth * 5.0, -1.0, 1.0))
        return np.array([angle, force, spin], dtype=np.float32)

    # Fallback toward goal
    gx, gy = env.layout.goal
    ang = math.atan2(gy - y, gx - x)
    return np.array([ang, 1.0, 0.0], dtype=np.float32)


def collect_augmented_transitions(cfg, dataset_path: str | Path, max_episodes: int = 3000):
    field = PoseGeodesicField(cfg.env.geodesic_field)
    z = np.load(dataset_path)
    offsets = z["offsets"]
    actions_flat = z["actions"]
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    aug_cfg = getattr(cfg.imitation, "augmented", None)
    xy_noise_std = float(getattr(aug_cfg, "xy_noise_std", 0.008)) if aug_cfg else 0.0
    ang_noise_std = float(getattr(aug_cfg, "angle_noise_std", 0.12)) if aug_cfg else 0.0
    noise_ratio = float(getattr(aug_cfg, "noise_ratio", 0.0)) if aug_cfg else 0.0

    cfg = cfg.copy()
    cfg.env.reward_mode = "sparse"
    env = AntSwarmEnv(config=cfg, seed=0)
    n_eps = min(len(offsets) - 1, max_episodes)
    logger.info(f"Collecting augmented transitions from {n_eps} demonstration episodes (noise_ratio={noise_ratio})...")

    all_obs, all_acts = [], []
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
            # 1. Clean transition
            obs = flatten(env.observation_space, env.obs_model.observe(env.state)).astype(np.float32)
            action = np.asarray(raw_action, dtype=np.float32).reshape(env.action_space.shape)
            all_obs.append(obs)
            all_acts.append(action.flatten())

            # 2. Synthetic perturbation transition
            if np.random.rand() < noise_ratio:
                orig_c = env.state.obj.center.copy()
                orig_a = float(env.state.obj.angle)
                
                # Apply small jitter
                perturbed_c = orig_c + np.random.normal(0, xy_noise_std, size=2).astype(np.float32)
                perturbed_a = orig_a + float(np.random.normal(0, ang_noise_std))
                
                env.state.reset(perturbed_c, perturbed_a)
                if not env.state._bad_pose():
                    pert_obs = flatten(env.observation_space, env.obs_model.observe(env.state)).astype(np.float32)
                    corrective_act = get_expert_action(field, env)
                    all_obs.append(pert_obs)
                    all_acts.append(corrective_act.flatten())
                
                # Restore state for forward simulation
                env.state.reset(orig_c, orig_a)

            next_raw, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if (i + 1) % 500 == 0 or (i + 1) == n_eps:
            logger.info(f"  Processed {i+1}/{n_eps} episodes ({len(all_obs)} total transitions, {time.time()-t0:.1f}s)")

    env.close()
    obs_tensor = torch.tensor(np.array(all_obs, dtype=np.float32), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(all_acts, dtype=np.float32), dtype=torch.float32)
    logger.info(f"Augmentation complete: {len(obs_tensor)} transitions generated.")
    return obs_tensor, act_tensor


def train_actor(
    actor: nn.Module,
    obs_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str = "cpu",
):
    actor.to(device)
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
            # In SB3 Actor, get action parameters directly for fast batch optimization
            mean_actions, _, _ = actor.get_action_dist_params(b_obs)
            # Reconstruct squashed action matching environment action bounds
            squashed = torch.tanh(mean_actions)
            # Rescale to action space: angle [-pi, pi], force [0, 1], spin [-1, 1]
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
        if ep % 5 == 0 or ep == epochs:
            logger.info(f"  Epoch {ep:2d}/{epochs:2d} - MSE Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    return best_loss


def evaluate_policy(model: SAC, cfg, n_episodes: int = 20, desc: str = "Evaluation"):
    z = np.load("storage_local/datasets/successes_v1/dataset.npz")
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=42))
    reach = float(cfg.goal.reach_radius)
    lengths, distances, successes = [], [], []

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
        done, steps = False, 0
        while not done and steps < 500:
            pred_a, _ = model.predict(obs, deterministic=True)
            action = np.asarray(pred_a, dtype=np.float32).reshape(eval_env.action_space.shape)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
            done = terminated or truncated

        dist = info.get("object_distance", base.state.distance_to_goal())
        succ = bool(info.get("is_success", False) or dist < reach)
        lengths.append(steps)
        distances.append(dist)
        successes.append(succ)

    eval_env.close()
    succ_rate = np.mean(successes) * 100.0
    mean_dist = np.mean(distances)
    logger.info(f"[{desc}] Success Rate={succ_rate:.1f}% ({np.sum(successes)}/{n_episodes}), Mean Final Dist={mean_dist:.4f}m")
    return succ_rate, mean_dist


def run_augmented_bc(cfg):
    logger.info("=== Starting Method 1: Perturbation-Augmented Imitation Learning ===")
    out_path = Path(cfg.imitation.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    obs_tensor, act_tensor = collect_augmented_transitions(
        cfg, cfg.imitation.dataset_path, max_episodes=int(cfg.imitation.episodes)
    )

    dummy_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
    sac_model = SAC("MlpPolicy", dummy_env, device="cpu", learning_rate=float(cfg.imitation.lr))
    
    train_actor(
        sac_model.actor,
        obs_tensor,
        act_tensor,
        epochs=int(cfg.imitation.epochs),
        batch_size=int(cfg.imitation.batch_size),
        lr=float(cfg.imitation.lr),
        device="cpu",
    )
    sac_model.save(out_path)
    logger.info(f"Saved Augmented BC model -> {out_path}")
    dummy_env.close()

    evaluate_policy(sac_model, cfg, n_episodes=20, desc="Augmented BC Final Eval")


def run_dagger(cfg):
    logger.info("=== Starting Method 2: DAgger (Dataset Aggregation) ===")
    field = PoseGeodesicField(cfg.env.geodesic_field)
    out_path = Path(cfg.imitation.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    dag_cfg = cfg.imitation.dagger
    n_iters = int(dag_cfg.iterations)
    rollout_eps = int(dag_cfg.rollout_episodes_per_iter)
    beta = float(dag_cfg.beta_init)
    beta_decay = float(dag_cfg.beta_decay)

    # Initial demonstration dataset
    obs_tensor, act_tensor = collect_augmented_transitions(
        cfg, cfg.imitation.dataset_path, max_episodes=int(cfg.imitation.init_episodes)
    )
    all_obs = [obs_tensor.numpy()]
    all_acts = [act_tensor.numpy()]

    dummy_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
    sac_model = SAC("MlpPolicy", dummy_env, device="cpu", learning_rate=float(cfg.imitation.lr))

    for iter_idx in range(n_iters):
        logger.info(f"\n--- DAgger Iteration {iter_idx+1}/{n_iters} (beta={beta:.2f}, total_samples={sum(len(x) for x in all_obs)}) ---")
        
        # Train on aggregated dataset
        curr_obs = torch.tensor(np.concatenate(all_obs, axis=0), dtype=torch.float32)
        curr_act = torch.tensor(np.concatenate(all_acts, axis=0), dtype=torch.float32)
        
        train_actor(
            sac_model.actor,
            curr_obs,
            curr_act,
            epochs=int(cfg.imitation.epochs_per_iter),
            batch_size=int(cfg.imitation.batch_size),
            lr=float(cfg.imitation.lr),
            device="cpu",
        )
        sac_model.save(out_path)

        # Rollout policy with expert queries
        new_obs, new_acts = [], []
        eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=iter_idx * 100))
        for ep in range(rollout_eps):
            obs, _ = eval_env.reset()
            done, steps = False, 0
            while not done and steps < 500:
                base = eval_env.unwrapped
                # Query ground-truth expert action for current state
                expert_act = get_expert_action(field, base)
                new_obs.append(obs.copy())
                new_acts.append(expert_act.flatten())

                # Mixed execution (beta-greedy)
                if np.random.rand() < beta:
                    act_exec = expert_act.reshape(eval_env.action_space.shape)
                else:
                    pred_a, _ = sac_model.predict(obs, deterministic=True)
                    act_exec = np.asarray(pred_a, dtype=np.float32).reshape(eval_env.action_space.shape)

                obs, reward, terminated, truncated, info = eval_env.step(act_exec)
                steps += 1
                done = terminated or truncated

        eval_env.close()
        all_obs.append(np.array(new_obs, dtype=np.float32))
        all_acts.append(np.array(new_acts, dtype=np.float32))
        logger.info(f"  Collected {len(new_obs)} interactive transitions from {rollout_eps} rollouts.")

        # Decay expert execution probability
        beta *= beta_decay
        evaluate_policy(sac_model, cfg, n_episodes=10, desc=f"DAgger Iter {iter_idx+1} Eval")

    dummy_env.close()
    evaluate_policy(sac_model, cfg, n_episodes=20, desc="DAgger Final 20-Ep Eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = getattr(cfg.imitation, "mode", "augmented_bc")

    if mode == "augmented_bc":
        run_augmented_bc(cfg)
    elif mode == "dagger":
        run_dagger(cfg)
    else:
        raise ValueError(f"Unknown imitation mode: {mode}")


if __name__ == "__main__":
    main()
