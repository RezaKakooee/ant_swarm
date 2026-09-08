import sys
from pathlib import Path
import numpy as np
from gymnasium.spaces.utils import flatten
from gymnasium.wrappers import FlattenObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/rl"))
from ant_swarm import AntSwarmEnv, load_config
from stable_baselines3 import SAC
from sac_bc import SAC_BC

cfg = load_config("configs/rl/gen_o_sac_bc.yaml")
cfg.env.reward_mode = "sparse"
z = np.load("storage_local/datasets/successes_v1/dataset.npz")
init_poses = z["init_pose"]
goals = z["goal"] if "goal" in z else None

ckpt_path = "storage_local/checkpoints/il_actor_best.zip"

print(f"Loading checkpoint: {ckpt_path}")
eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=42))
reach = float(cfg.goal.reach_radius)

model = SAC.load(
    ckpt_path,
    env=eval_env,
    custom_objects={"observation_space": eval_env.observation_space, "action_space": eval_env.action_space},
)

# Test A: Demo Initial Poses
demo_succ, demo_dists = [], []
for i in range(20):
    eval_env.reset()
    base = eval_env.unwrapped
    if goals is not None:
        base.layout.goal = np.asarray(goals[i], dtype=np.float32)
    center = np.asarray(init_poses[i][:2], dtype=np.float32)
    angle = float(init_poses[i][2])
    base.state.reset(center, angle)
    if base.state._bad_pose():
        continue
    base.init_center, base.init_angle = center.copy(), angle
    base.reward_model.reset(base.state)

    obs = flatten(base.observation_space, base.obs_model.observe(base.state)).astype(np.float32)
    done, steps = False, 0
    while not done and steps < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        steps += 1
        done = terminated or truncated

    dist = info.get("object_distance", base.state.distance_to_goal())
    succ = bool(info.get("is_success", False) or dist < reach)
    demo_succ.append(succ)
    demo_dists.append(dist)
    print(f"Demo Init Pose {i+1:02d}: FinalDist={dist:.4f}m, Steps={steps}, Success={succ}")

print(f"\n[A] SAC+BC on Demo Poses (20 eps): Success Rate = {np.mean(demo_succ)*100:.1f}%, Mean Final Dist = {np.mean(demo_dists):.4f}m")

# Test B: Random Full Maze
rand_succ, rand_dists = [], []
for i in range(20):
    obs, _ = eval_env.reset()
    done, steps = False, 0
    while not done and steps < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        steps += 1
        done = terminated or truncated
    dist = info.get("object_distance", 999.0)
    succ = bool(info.get("is_success", False) or dist < reach)
    rand_succ.append(succ)
    rand_dists.append(dist)

print(f"[B] SAC+BC on Random Maze (20 eps): Success Rate = {np.mean(rand_succ)*100:.1f}%, Mean Final Dist = {np.mean(rand_dists):.4f}m")
eval_env.close()
