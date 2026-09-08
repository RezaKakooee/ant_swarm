"""Test waypoint-conditioned closed-loop policy execution."""
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
goals = z["goal"]

reach = float(cfg.goal.reach_radius)
print(f"=== Waypoint-Conditioned Closed-Loop Policy Evaluation (20 Episodes) ===")

successes = []
final_dists = []

for ep in range(20):
    env = FlattenObservation(AntSwarmEnv(config=cfg, seed=ep))
    env.reset()
    base = env.unwrapped
    
    final_goal = np.asarray(goals[ep], dtype=np.float32)
    center = np.asarray(init_poses[ep][:2], dtype=np.float32)
    angle = float(init_poses[ep][2])
    base.state.reset(center, angle)
    base.init_center, base.init_angle = center.copy(), angle
    base.reward_model.reset(base.state)

    # Waypoint 1: Slit 1 center (0.758, 0.36)
    # Waypoint 2: Slit 2 center (0.979, 0.36)
    # Waypoint 3: Final goal
    waypoints = [
        np.array([0.758, 0.36], dtype=np.float32),
        np.array([0.979, 0.36], dtype=np.float32),
        final_goal,
    ]
    wp_idx = 0
    base.layout.goal = waypoints[wp_idx].copy()
    base.obs_model.goal = waypoints[wp_idx].copy()

    obs = flatten(base.observation_space, base.obs_model.observe(base.state)).astype(np.float32)
    hit_final = False

    for t in range(400):
        # Check waypoint advance
        cur_x = base.state.obj.center[0]
        if wp_idx == 0 and cur_x > 0.77:
            wp_idx = 1
            base.layout.goal = waypoints[wp_idx].copy()
            base.obs_model.goal = waypoints[wp_idx].copy()
        elif wp_idx == 1 and cur_x > 0.99:
            wp_idx = 2
            base.layout.goal = waypoints[wp_idx].copy()
            base.obs_model.goal = waypoints[wp_idx].copy()

        pred_act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(pred_act.reshape(env.action_space.shape))
        
        dist_to_final = np.linalg.norm(base.state.obj.center - final_goal)
        if dist_to_final < reach:
            hit_final = True
            break
        if term:
            break

    final_d = np.linalg.norm(base.state.obj.center - final_goal)
    successes.append(hit_final or final_d < reach)
    final_dists.append(final_d)
    print(f"Ep {ep+1:02d}: Final Dist={final_d:.4f}m | Success={hit_final or final_d < reach} (Wp reached: {wp_idx+1}/3)")
    env.close()

succ_rate = np.mean(successes) * 100.0
mean_d = np.mean(final_dists)
print("-" * 75)
print(f"WAYPOINT EVALUATION RESULTS: Success Rate = {succ_rate:.1f}% ({np.sum(successes)}/20) | Mean Final Dist = {mean_d:.4f}m")
