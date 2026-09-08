import numpy as np

z = np.load("storage_local/datasets/successes_v1/dataset.npz")
print("Keys:", list(z.keys()))
offsets = z["offsets"]
actions = z["actions"]
init_pose = z["init_pose"]
goals = z["goal"] if "goal" in z else None

print(f"Trajectories: {len(offsets)-1}, Total transitions: {len(actions)}")
print(f"Lengths min/mean/max: {np.diff(offsets).min()}/{np.diff(offsets).mean():.1f}/{np.diff(offsets).max()}")
print(f"Init x min/mean/max: {init_pose[:,0].min():.3f}/{init_pose[:,0].mean():.3f}/{init_pose[:,0].max():.3f}")
print(f"Init y min/mean/max: {init_pose[:,1].min():.3f}/{init_pose[:,1].mean():.3f}/{init_pose[:,1].max():.3f}")
print(f"Init theta min/mean/max: {init_pose[:,2].min():.3f}/{init_pose[:,2].mean():.3f}/{init_pose[:,2].max():.3f}")
if goals is not None:
    print(f"Goals min/max: x in [{goals[:,0].min():.3f}, {goals[:,0].max():.3f}], y in [{goals[:,1].min():.3f}, {goals[:,1].max():.3f}]")

with open("storage_local/dataset_info.txt", "w") as f:
    f.write(f"Trajectories: {len(offsets)-1}\n")
    f.write(f"Init x: min={init_pose[:,0].min():.3f}, max={init_pose[:,0].max():.3f}\n")
    f.write(f"Init y: min={init_pose[:,1].min():.3f}, max={init_pose[:,1].max():.3f}\n")
    f.write(f"Init theta: min={init_pose[:,2].min():.3f}, max={init_pose[:,2].max():.3f}\n")
    if goals is not None:
        f.write(f"Goal: x=[{goals[:,0].min():.3f}, {goals[:,0].max():.3f}], y=[{goals[:,1].min():.3f}, {goals[:,1].max():.3f}]\n")
print("Done writing dataset_info.txt")
