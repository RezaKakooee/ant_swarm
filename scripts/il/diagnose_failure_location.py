"""Where does the closed-loop rollout actually die?

Every method tried so far lands at ~0.59 mean final distance with 0% success.
Before trying another method, measure the failure: how far along the maze the
load gets, whether it clears each slit, and whether it jams (stops moving) or
wanders.  Pure diagnosis -- no field, no training.

    python scripts/il/diagnose_failure_location.py \
        --ckpt storage_local/checkpoints/bc_chunk_H1.pt --episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces.utils import flatten
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "il"))
from ant_swarm import AntSwarmEnv, load_config          # noqa: E402
from train_chunked_bc import ChunkPolicy, pick_eval_episodes  # noqa: E402

STUCK_WINDOW = 100          # steps
STUCK_TRAVEL = 0.01         # metres moved within the window -> jammed


@torch.no_grad()
def rollout(net, env, init_pose, goal, max_steps, dev, tip_local):
    env.reset()
    env.layout.goal = np.asarray(goal, dtype=np.float32)
    center = np.asarray(init_pose[:2], dtype=np.float32)
    angle = float(init_pose[2])
    env.state.reset(center, angle)
    if env.state._bad_pose():
        return None
    env.init_center, env.init_angle = center.copy(), angle
    env.reward_model.reset(env.state)

    xs, ys, tipx, ang, info, done, steps = [], [], [], [], {}, False, 0
    while not done and steps < max_steps:
        o = flatten(env.observation_space,
                    env.obs_model.observe(env.state)).astype(np.float32)
        a = net(torch.tensor(o[None, :], device=dev)).cpu().numpy()[0][0]
        _, _, term, trunc, info = env.step(
            np.asarray(a, dtype=np.float32).reshape(env.action_space.shape))
        obj = env.state.obj
        xs.append(float(obj.center[0]))
        ys.append(float(obj.center[1]))
        # landmark on the LEADING TIP, not the centre: the T is 0.33 m long,
        # so its centre sits ~0.17 m behind the part that must clear the wall
        tips = obj.center[None, :] + tip_local @ obj.rot().T
        tipx.append(float(tips[:, 0].max()))
        ang.append(float(obj.angle))
        steps += 1
        done = term or trunc
    return (np.asarray(xs), np.asarray(ys), np.asarray(tipx),
            np.asarray(ang), info, steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    p.add_argument("--ckpt", default="storage_local/checkpoints/bc_chunk_H1.pt")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    cfg = load_config(args.config)
    s = float(cfg.scene_scale)
    slit1, slit2 = [float(v) * s for v in cfg.walls.x_columns]
    reach = float(cfg.goal.reach_radius) * s

    blob = torch.load(args.ckpt, map_location=dev)
    env = AntSwarmEnv(config=cfg, seed=42)
    obs_dim = env.observation_space.shape[-1] * env.observation_space.shape[0] \
        if len(env.observation_space.shape) > 1 else env.observation_space.shape[0]
    net = ChunkPolicy(obs_dim, blob.get("horizon", 1)).to(dev)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    logger.info(f"loaded {args.ckpt} (horizon={blob.get('horizon', 1)}, obs_dim={obs_dim})")

    z = np.load(args.dataset)
    init_poses, goals = z["init_pose"], z["goal"]

    eval_eps = pick_eval_episodes(init_poses, args.episodes)
    logger.info(f"{len(eval_eps)} eval episodes with DISTINCT start poses")
    rows = []
    for ep in eval_eps:
        ep = int(ep)
        out = rollout(net, env, init_poses[ep], goals[ep], args.max_steps,
                      dev, env.obs_model.tip_local)
        if out is None:
            continue
        xs, ys, tipx, ang, info, steps = out
        travel = float(np.abs(xs[-1] - xs[max(0, len(xs) - STUCK_WINDOW)]))
        travel += float(np.abs(ys[-1] - ys[max(0, len(ys) - STUCK_WINDOW)]))
        rows.append(dict(
            max_x=float(xs.max()), max_tip=float(tipx.max()),
            final_tip=float(tipx[-1]), final_angle=float(ang[-1]),
            final_x=float(xs[-1]), final_y=float(ys[-1]),
            dist=float(info.get("object_distance", env.state.distance_to_goal())),
            success=bool(info.get("is_success", False)),
            steps=steps, jammed=travel < STUCK_TRAVEL))
    env.close()

    n = len(rows)
    max_x = np.array([r["max_x"] for r in rows])
    max_tip = np.array([r["max_tip"] for r in rows])
    fin_ang = np.array([r["final_angle"] for r in rows])
    final_x = np.array([r["final_x"] for r in rows])
    dist = np.array([r["dist"] for r in rows])
    jammed = np.array([r["jammed"] for r in rows])

    # how far each episode got, by landmark
    stages = [
        ("tip never reached slit 1", max_tip < slit1),
        ("tip touched slit 1, stuck", (max_tip >= slit1) & (max_x < slit1)),
        ("centre through slit 1", (max_x >= slit1) & (max_x < slit2)),
        ("centre through slit 2", max_x >= slit2),
    ]

    print("\n" + "=" * 66)
    print(f"FAILURE LOCATION  ({n} episodes, {args.ckpt})")
    print(f"slit 1 x={slit1:.3f}   slit 2 x={slit2:.3f}   goal reach={reach:.3f}")
    print("=" * 66)
    print(f"\n{'furthest the load got':<26} {'episodes':>9} {'%':>7} {'mean final dist':>17}")
    print("-" * 66)
    for label, m in stages:
        k = int(m.sum())
        md = float(dist[m].mean()) if k else float("nan")
        print(f"{label:<26} {k:>9} {100*k/n:>6.1f}% {md:>17.4f}")
    print(f"\n{'jammed at the end (moved <10 mm in last 100 steps)':<52} "
          f"{int(jammed.sum())}/{n}")
    print(f"{'reached the goal':<52} {sum(r['success'] for r in rows)}/{n}")
    # the big cap (0.1748 m) is wider than the 0.150 m gap, so the load can only
    # thread the slit while tilted -- never lying parallel to the wall plane
    tilt = np.abs(np.degrees(np.arctan2(np.sin(fin_ang), np.cos(fin_ang))))
    print(f"\nfinal |angle| (deg): mean={tilt.mean():.1f}  median={np.median(tilt):.1f}")
    near_par = int((np.minimum(np.abs(tilt - 90), np.abs(tilt - 270)) < 20).sum())
    print(f"jammed lying near-parallel to the wall (|angle-90| < 20 deg): {near_par}/{n}")
    print(f"max tip x: mean={max_tip.mean():.4f}  median={np.median(max_tip):.4f}  "
          f"max={max_tip.max():.4f}   (slit 1 wall at {slit1:.3f})")
    print(f"max x   : mean={max_x.mean():.4f}  median={np.median(max_x):.4f}  "
          f"max={max_x.max():.4f}")
    print(f"final x : mean={final_x.mean():.4f}  median={np.median(final_x):.4f}")
    print(f"final d : mean={dist.mean():.4f}  median={np.median(dist):.4f}  "
          f"min={dist.min():.4f}\n")


if __name__ == "__main__":
    main()
