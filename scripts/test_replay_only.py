import sys
import numpy as np

sys.path.insert(0, ".")
from ant_swarm import AntSwarmEnv, load_config

cfg = load_config("configs/rl/gen_m_demo_seeded.yaml")
cfg.env.reward_mode = "sparse"
z = np.load("storage_local/datasets/successes_v1/dataset.npz")
offsets = z["offsets"]
actions_flat = z["actions"]
init_poses = z["init_pose"]
goals = z["goal"] if "goal" in z else None

env = AntSwarmEnv(config=cfg, seed=0)
reach = float(cfg.goal.reach_radius)
successes = []
final_dists = []

for i in range(20):
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

    for act in actions:
        obs, reward, terminated, truncated, info = env.step(act.reshape(env.action_space.shape))
        if terminated or truncated:
            break

    dist = info.get("object_distance", env.state.distance_to_goal())
    succ = bool(info.get("is_success", False) or dist < reach)
    successes.append(succ)
    final_dists.append(dist)
    print(f"Demo {i+1:02d}: Steps={len(actions)}, FinalDist={dist:.4f}m, Success={succ}")

print(f"\nSummary (20 demos): Success Rate = {np.mean(successes)*100:.1f}%, Mean Final Dist = {np.mean(final_dists):.4f}m")
