"""Check minimum distance to goal and goal crossing during episode rollouts."""
from __future__ import annotations

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
init_poses = z["init_pose"]
goals = z["goal"] if "goal" in z else None

reach = float(cfg.goal.reach_radius)
print(f"=== Policy Goal Crossing Analysis (Goal Reach Radius = {reach}m) ===")

reached_goal_at_any_step = []
min_distances = []
final_distances = []

for ep in range(20):
    env = FlattenObservation(AntSwarmEnv(config=cfg, seed=ep))
    env.reset()
    base = env.unwrapped
    
    if goals is not None:
        base.layout.goal = np.asarray(goals[ep], dtype=np.float32)
    center = np.asarray(init_poses[ep][:2], dtype=np.float32)
    angle = float(init_poses[ep][2])
    base.state.reset(center, angle)
    base.init_center, base.init_angle = center.copy(), angle
    base.reward_model.reset(base.state)

    obs = flatten(base.observation_space, base.obs_model.observe(base.state)).astype(np.float32)
    
    min_d = float("inf")
    hit_goal = False
    hit_step = -1
    max_x = -float("inf")

    for t in range(300):
        pred_act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(pred_act.reshape(env.action_space.shape))
        
        cur_pos = base.state.obj.center
        max_x = max(max_x, cur_pos[0])
        dist = base.state.distance_to_goal()
        if dist < min_d:
            min_d = dist
        if dist < reach and not hit_goal:
            hit_goal = True
            hit_step = t
        if term:
            break

    final_d = base.state.distance_to_goal()
    reached_goal_at_any_step.append(hit_goal)
    min_distances.append(min_d)
    final_distances.append(final_d)
    
    status = f"HIT GOAL at step {hit_step:03d}!" if hit_goal else f"Missed (Min Dist: {min_d:.4f}m)"
    print(f"Ep {ep+1:02d}: Max X={max_x:.3f}m | Final Dist={final_d:.4f}m | {status}")
    env.close()

hit_rate = np.mean(reached_goal_at_any_step) * 100.0
mean_min_d = np.mean(min_distances)
mean_fin_d = np.mean(final_distances)
print("-" * 75)
print(f"SUMMARY: Goal Touch Rate: {hit_rate:.1f}% ({np.sum(reached_goal_at_any_step)}/20)")
print(f"Mean Min Distance to Goal: {mean_min_d:.4f}m | Mean Final Dist: {mean_fin_d:.4f}m")
