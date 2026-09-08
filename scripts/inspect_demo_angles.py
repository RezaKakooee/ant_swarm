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

env = AntSwarmEnv(config=cfg, seed=0)

print("Inspecting Demo 01 Trajectory Poses:")
actions = actions_flat[offsets[0]:offsets[1]]
init_pose = init_poses[0]
env.reset()
center = np.asarray(init_pose[:2], dtype=np.float32)
angle = float(init_pose[2])
env.state.reset(center, angle)

slit1_x = 0.758
slit2_x = 0.979

for t, act in enumerate(actions):
    obs, r, term, trunc, info = env.step(act.reshape(env.action_space.shape))
    pos = env.state.obj.center
    ang = env.state.obj.angle
    if abs(pos[0] - slit1_x) < 0.05 or abs(pos[0] - slit2_x) < 0.05 or t % 30 == 0:
        print(f"Step {t:03d}: x={pos[0]:.3f}, y={pos[1]:.3f}, angle={np.degrees(ang):.1f}° (rad={ang:.3f})")
    if term or trunc:
        break
