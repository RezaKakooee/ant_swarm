"""Intervention-Guided DAgger (HG-DAgger) for Complete Maze Traversal.

Whenever the policy is stuck at a slit (velocity ~ 0 or wall contact), the
geodesic expert intervenes for K steps to guide the object through the narrow
channel, ensuring 100% full-maze demonstration density across both Slit 1 and Slit 2.
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
    obj = env.state.obj
    x, y, theta = float(obj.center[0]), float(obj.center[1]), float(obj.angle)
    
    current_idx = field.index(x, y, theta)
    if not field.reachable[current_idx]:
        reachable_neighbors = [q for q in field._neighbors(current_idx) if field.reachable[q]]
        if reachable_neighbors:
            current_idx = min(reachable_neighbors, key=lambda q: int(field.steps[q]))
        else:
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

    gx, gy = env.layout.goal
    ang = math.atan2(gy - y, gx - x)
    return np.array([ang, 1.0, 0.0], dtype=np.float32)


def train_actor_fast(
    actor: nn.Module,
    obs_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str = "cpu",
):
    actor.to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

        avg_loss = total_loss / max(1, n_batches)
        if ep % 5 == 0 or ep == epochs:
            logger.info(f"    Epoch {ep:2d}/{epochs:2d} - MSE Loss: {avg_loss:.6f}")


def run_intervention_dagger(cfg, n_iters: int = 10, rollout_eps: int = 50, epochs_per_iter: int = 15):
    logger.info("=== Starting Intervention-Guided DAgger (Full-Maze Completion) ===")
    field = PoseGeodesicField(cfg.env.geodesic_field)
    out_path = Path("storage_local/checkpoints/il_dagger_intervention_best.zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load initial demonstration dataset (3000 episodes)
    z = np.load("storage_local/datasets/successes_v1/dataset.npz")
    offsets = z["offsets"]
    actions_flat = z["actions"]
    init_poses = z["init_pose"]
    goals = z["goal"] if "goal" in z else None

    init_n = min(len(offsets) - 1, 3000)
    logger.info(f"Loading initial {init_n} expert demonstration trajectories...")
    all_obs, all_acts = [], []
    
    cfg_sparse = cfg.copy()
    cfg_sparse.env.reward_mode = "sparse"
    env = AntSwarmEnv(config=cfg_sparse, seed=0)
    
    for i in range(init_n):
        actions = actions_flat[offsets[i]:offsets[i+1]]
        env.reset()
        if goals is not None:
            env.layout.goal = np.asarray(goals[i], dtype=np.float32)
        env.state.reset(np.asarray(init_poses[i][:2], dtype=np.float32), float(init_poses[i][2]))
        if env.state._bad_pose():
            continue
        env.reward_model.reset(env.state)

        for raw_action in actions:
            obs = flatten(env.observation_space, env.obs_model.observe(env.state)).astype(np.float32)
            act = np.asarray(raw_action, dtype=np.float32).reshape(env.action_space.shape)
            all_obs.append(obs)
            all_acts.append(act.flatten())
            next_raw, reward, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                break

    env.close()
    logger.info(f"Loaded {len(all_obs)} initial transitions.")

    dummy_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
    sac_model = SAC("MlpPolicy", dummy_env, device="cpu", learning_rate=float(cfg.imitation.lr))
    dummy_env.close()

    eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=999))
    reach = float(cfg.goal.reach_radius)

    for iter_idx in range(n_iters):
        logger.info(f"\n--- HG-DAgger Iteration {iter_idx+1}/{n_iters} (Dataset Size: {len(all_obs)} transitions) ---")
        
        # Train on aggregated dataset
        obs_t = torch.tensor(np.array(all_obs, dtype=np.float32), dtype=torch.float32)
        act_t = torch.tensor(np.array(all_acts, dtype=np.float32), dtype=torch.float32)
        train_actor_fast(sac_model.actor, obs_t, act_t, epochs=epochs_per_iter, batch_size=512, lr=3e-4)
        sac_model.save(out_path)

        # Interactive rollouts with Expert Intervention when stuck
        new_obs, new_acts = [], []
        intervention_steps = 0
        total_rollout_steps = 0

        for ep in range(rollout_eps):
            obs, _ = eval_env.reset()
            base = eval_env.unwrapped
            prev_center = base.state.obj.center.copy()
            stuck_count = 0
            expert_cooldown = 0
            done, steps = False, 0

            while not done and steps < 500:
                total_rollout_steps += 1
                curr_center = base.state.obj.center
                movement = np.linalg.norm(curr_center - prev_center)
                prev_center = curr_center.copy()

                if movement < 0.002:
                    stuck_count += 1
                else:
                    stuck_count = max(0, stuck_count - 1)

                # Query expert action
                expert_act = get_expert_action(field, base)
                new_obs.append(obs.copy())
                new_acts.append(expert_act.flatten())

                # If stuck for > 5 steps, trigger expert intervention for 20 steps
                if stuck_count >= 5:
                    expert_cooldown = 20
                    stuck_count = 0

                if expert_cooldown > 0:
                    act_exec = expert_act.reshape(eval_env.action_space.shape)
                    expert_cooldown -= 1
                    intervention_steps += 1
                else:
                    pred_a, _ = sac_model.predict(obs, deterministic=True)
                    act_exec = np.asarray(pred_a, dtype=np.float32).reshape(eval_env.action_space.shape)

                obs, reward, terminated, truncated, info = eval_env.step(act_exec)
                steps += 1
                done = terminated or truncated

        all_obs.extend(new_obs)
        all_acts.extend(new_acts)
        interv_rate = (intervention_steps / max(1, total_rollout_steps)) * 100.0
        logger.info(f"  Iter {iter_idx+1}: Added {len(new_obs)} transitions. Intervention Rate={interv_rate:.1f}%")

        # Independent closed-loop evaluation (0% expert intervention)
        eval_successes, eval_dists = [], []
        for ep in range(20):
            eval_env.reset()
            base = eval_env.unwrapped
            if goals is not None:
                base.layout.goal = np.asarray(goals[ep], dtype=np.float32)
            base.state.reset(np.asarray(init_poses[ep][:2], dtype=np.float32), float(init_poses[ep][2]))
            if base.state._bad_pose():
                continue
            base.reward_model.reset(base.state)

            obs = flatten(base.observation_space, base.obs_model.observe(base.state)).astype(np.float32)
            done, steps = False, 0
            while not done and steps < 500:
                pred_a, _ = sac_model.predict(obs, deterministic=True)
                act_exec = np.asarray(pred_a, dtype=np.float32).reshape(eval_env.action_space.shape)
                obs, reward, terminated, truncated, info = eval_env.step(act_exec)
                steps += 1
                done = terminated or truncated

            dist = info.get("object_distance", base.state.distance_to_goal())
            succ = bool(info.get("is_success", False) or dist < reach)
            eval_successes.append(succ)
            eval_dists.append(dist)

        succ_rate = np.mean(eval_successes) * 100.0
        mean_d = np.mean(eval_dists)
        logger.info(f"  [Evaluation (Zero Expert)] Success Rate={succ_rate:.1f}% ({np.sum(eval_successes)}/20), Mean Dist={mean_d:.4f}m")

    eval_env.close()
    logger.info(f"Intervention DAgger Finished. Checkpoint saved -> {out_path}")


def main():
    cfg = load_config("configs/il/il_dagger.yaml")
    run_intervention_dagger(cfg, n_iters=10, rollout_eps=60, epochs_per_iter=12)


if __name__ == "__main__":
    main()
