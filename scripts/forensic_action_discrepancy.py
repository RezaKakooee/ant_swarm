"""Forensic comparison of open-loop expert action execution vs closed-loop neural policy."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from gymnasium.spaces.utils import flatten
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rl"))
from ant_swarm import AntSwarmEnv, load_config
from scripts.rl.sac_bc import SAC_BC

cfg = load_config("configs/rl/gen_n_bc_transfer.yaml")
model_path = "storage_local/checkpoints/il_full_dataset_best.zip"

dummy_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
model = SAC.load(model_path, env=dummy_env, device="cpu")
dummy_env.close()

z = np.load("storage_local/datasets/successes_v1/dataset.npz")
offsets = z["offsets"]
actions_flat = z["actions"]
init_poses = z["init_pose"]
goals = z["goal"] if "goal" in z else None

# Compare on Episode 0
expert_actions = actions_flat[offsets[0]:offsets[0+1]]
init_pose = init_poses[0]
goal = goals[0] if goals is not None else None

print("=== Forensic Action Discrepancy Analysis (Episode 0) ===")
print(f"Init Pose: (x={init_pose[0]:.3f}, y={init_pose[1]:.3f}, th={np.degrees(init_pose[2]):.1f}°)")

# 1. Closed-Loop Rollout with Policy
env_policy = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
env_policy.reset()
base_p = env_policy.unwrapped
if goal is not None:
    base_p.layout.goal = np.asarray(goal, dtype=np.float32)
base_p.state.reset(np.asarray(init_pose[:2], dtype=np.float32), float(init_pose[2]))
base_p.init_center, base_p.init_angle = np.asarray(init_pose[:2], dtype=np.float32), float(init_pose[2])
base_p.reward_model.reset(base_p.state)

# 2. Open-Loop Replay with Expert
env_expert = FlattenObservation(AntSwarmEnv(config=cfg, seed=0))
env_expert.reset()
base_e = env_expert.unwrapped
if goal is not None:
    base_e.layout.goal = np.asarray(goal, dtype=np.float32)
base_e.state.reset(np.asarray(init_pose[:2], dtype=np.float32), float(init_pose[2]))
base_e.init_center, base_e.init_angle = np.asarray(init_pose[:2], dtype=np.float32), float(init_pose[2])
base_e.reward_model.reset(base_e.state)

print(f"{'Step':<5} | {'Exp Act [ang°, frc, spn]':<25} | {'Pol Act [ang°, frc, spn]':<25} | {'Exp Pose (x, y, th°)':<25} | {'Pol Pose (x, y, th°)':<25}")
print("-" * 115)

obs_p = flatten(base_p.observation_space, base_p.obs_model.observe(base_p.state)).astype(np.float32)

for t in range(min(len(expert_actions), 150)):
    exp_act = np.asarray(expert_actions[t], dtype=np.float32).flatten()
    pred_act, _ = model.predict(obs_p, deterministic=True)
    pred_act = np.asarray(pred_act, dtype=np.float32).flatten()
    
    # Step expert
    obs_e, r_e, term_e, trunc_e, info_e = env_expert.step(exp_act.reshape(env_expert.action_space.shape))
    
    # Step policy
    obs_p, r_p, term_p, trunc_p, info_p = env_policy.step(pred_act.reshape(env_policy.action_space.shape))

    pos_e = base_e.state.obj.center
    ang_e = np.degrees(base_e.state.obj.angle)
    pos_p = base_p.state.obj.center
    ang_p = np.degrees(base_p.state.obj.angle)

    if t % 10 == 0 or t in [70, 75, 80, 85, 90, 95, 100]:
        exp_a_str = f"[{np.degrees(exp_act[0]):6.1f}°, {exp_act[1]:4.2f}, {exp_act[2]:5.2f}]"
        pol_a_str = f"[{np.degrees(pred_act[0]):6.1f}°, {pred_act[1]:4.2f}, {pred_act[2]:5.2f}]"
        exp_p_str = f"({pos_e[0]:.3f}, {pos_e[1]:.3f}, {ang_e:6.1f}°)"
        pol_p_str = f"({pos_p[0]:.3f}, {pos_p[1]:.3f}, {ang_p:6.1f}°)"
        print(f"{t:<5} | {exp_a_str:<25} | {pol_a_str:<25} | {exp_p_str:<25} | {pol_p_str:<25}")

print(f"\nFinal Expert Dist: {info_e.get('object_distance'):.4f}m (Success: {info_e.get('is_success')})")
print(f"Final Policy Dist: {info_p.get('object_distance'):.4f}m (Success: {info_p.get('is_success')})")

env_expert.close()
env_policy.close()
