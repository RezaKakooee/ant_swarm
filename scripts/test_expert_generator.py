"""Test geodesic expert demonstration generator across random starting poses."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from ant_swarm import AntSwarmEnv, load_config
from ant_swarm.geodesic import PoseGeodesicField
from scripts.il.train_hg_dagger import get_expert_action

cfg = load_config("configs/il/il_augmented_bc.yaml")
field = PoseGeodesicField(cfg.env.geodesic_field)

env = AntSwarmEnv(config=cfg, seed=42)
reach = float(cfg.goal.reach_radius)

print("Testing closed-loop expert trajectory generation across random initial poses...")
successes = 0
n_test = 20

t0 = time.time()
for ep in range(n_test):
    env.reset()
    done, steps = False, 0
    while not done and steps < 500:
        action = get_expert_action(field, env.unwrapped)
        obs, reward, terminated, truncated, info = env.step(action.reshape(env.action_space.shape))
        steps += 1
        done = terminated or truncated

    dist = env.unwrapped.state.distance_to_goal()
    succ = dist < reach
    if succ:
        successes += 1
    print(f"Episode {ep+1:02d}: Start=(x={env.init_center[0]:.2f}, y={env.init_center[1]:.2f}, th={np.degrees(env.init_angle):.0f}°), Steps={steps:03d}, Final Dist={dist:.4f}m, Success={succ}")

elapsed = time.time() - t0
print(f"\nExpert Success Rate: {successes}/{n_test} ({successes/n_test*100:.1f}%) in {elapsed:.2f}s ({n_test/elapsed:.1f} eps/sec)")
