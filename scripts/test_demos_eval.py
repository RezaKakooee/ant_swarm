import sys
from pathlib import Path
import numpy as np
from gymnasium.spaces.utils import flatten
from gymnasium.wrappers import FlattenObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from ant_swarm import AntSwarmEnv, load_config
from stable_baselines3 import SAC

cfg = load_config("configs/rl/gen_m_demo_seeded.yaml")
cfg.env.reward_mode = "sparse"
z = np.load("storage_local/datasets/successes_v1/dataset.npz")
offsets = z["offsets"]
actions_flat = z["actions"]
init_poses = z["init_pose"]
goals = z["goal"] if "goal" in z else None

# Test 1: Open-loop replaying recorded demo actions in the environment
env = AntSwarmEnv(config=cfg, seed=0)
reach = float(cfg.goal.reach_radius)
replay_succ = []
final_dists = []

for i in range(5):
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

    for act in actions:
        obs, reward, terminated, truncated, info = env.step(act.reshape(env.action_space.shape))
        if terminated or truncated:
            break

    dist = info.get("object_distance", env.state.distance_to_goal())
    succ = bool(info.get("is_success", False) or dist < reach)
    replay_succ.append(succ)
    final_dists.append(dist)

env.close()
print(f"[1] Open-Loop Demo Replay (50 eps): Success Rate={np.mean(replay_succ)*100:.1f}%, Mean Final Dist={np.mean(final_dists):.4f}m")

# Test 2: Evaluate BC Actor Model
bc_path = "storage_local/checkpoints/bc_actor_model.zip"
eval_env = FlattenObservation(AntSwarmEnv(config=cfg, seed=42))
bc_model = SAC.load(
    bc_path,
    env=eval_env,
    custom_objects={"observation_space": eval_env.observation_space, "action_space": eval_env.action_space},
)

bc_demo_succ, bc_demo_dists = [], []
for i in range(5):
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
    done = False
    steps = 0
    while not done and steps < 500:
        action, _ = bc_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        steps += 1
        done = terminated or truncated

    dist = info.get("object_distance", base.state.distance_to_goal())
    succ = bool(info.get("is_success", False) or dist < reach)
    bc_demo_succ.append(succ)
    bc_demo_dists.append(dist)

print(f"[2] BC Model on DEMO Init Poses (50 eps): Success Rate={np.mean(bc_demo_succ)*100:.1f}%, Mean Final Dist={np.mean(bc_demo_dists):.4f}m")

# Test 3: Evaluate BC Model on Random Maze Spawns
bc_rand_succ, bc_rand_dists = [], []
for _ in range(50):
    obs, _ = eval_env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        action, _ = bc_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        steps += 1
        done = terminated or truncated
    dist = info.get("object_distance", 999.0)
    succ = bool(info.get("is_success", False) or dist < reach)
    bc_rand_succ.append(succ)
    bc_rand_dists.append(dist)

print(f"[3] BC Model on RANDOM Maze Spawns (50 eps): Success Rate={np.mean(bc_rand_succ)*100:.1f}%, Mean Final Dist={np.mean(bc_rand_dists):.4f}m")
eval_env.close()
