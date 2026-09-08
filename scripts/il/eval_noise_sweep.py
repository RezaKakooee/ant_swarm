"""Does action noise break the deterministic deadlock?

Diagnosis that motivates this
-----------------------------
Rolling out the trained BC policy, every episode ends the same way: the load
wedges, the observation stops changing, and the policy -- being deterministic --
returns the SAME action forever.  Measured over the last 100 steps of an
episode the action standard deviation is exactly 0.00 and the speed is exactly
0.000000.  That is a fixed point, not a control failure: pi(o) -> a -> o, over
and over.

The expert demos contain only successes, so BC never saw a wedged state and has
no escape.  But any perturbation breaks a fixed point.  This sweeps Gaussian
action noise to see whether the deadlock is the thing capping success at 0%.

Note this runs opposite to handoff root cause #4, which blamed SAC's noise for
causing jams.  Here noise is tested as the cure, not the disease.

    python scripts/il/eval_noise_sweep.py --ckpt storage_local/checkpoints/bc_chunk_H1.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces.utils import flatten
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "il"))
from ant_swarm import AntSwarmEnv, load_config                       # noqa: E402
from train_chunked_bc import ChunkPolicy, pick_eval_episodes         # noqa: E402

STUCK_WINDOW, STUCK_TRAVEL = 100, 0.01


@torch.no_grad()
def run(net, cfg, init_poses, goals, eval_eps, noise, mode, dev,
        max_steps=500, seed=42):
    """mode: 'always' = noise every step; 'stuck' = only once wedged."""
    env = AntSwarmEnv(config=cfg, seed=seed)
    reach = float(cfg.goal.reach_radius) * float(cfg.scene_scale)
    rng = np.random.default_rng(0)
    dists, succ, jam = [], [], []

    for ep in eval_eps:
        ep = int(ep)
        env.reset()
        env.layout.goal = np.asarray(goals[ep], dtype=np.float32)
        c = np.asarray(init_poses[ep][:2], dtype=np.float32)
        a0 = float(init_poses[ep][2])
        env.state.reset(c, a0)
        if env.state._bad_pose():
            continue
        env.init_center, env.init_angle = c.copy(), a0
        env.reward_model.reset(env.state)

        info, done, steps, hist = {}, False, 0, []
        while not done and steps < max_steps:
            o = flatten(env.observation_space,
                        env.obs_model.observe(env.state)).astype(np.float32)
            a = net(torch.tensor(o[None, :], device=dev)).cpu().numpy()[0][0].copy()

            wedged = False
            if len(hist) >= STUCK_WINDOW:
                wedged = float(np.linalg.norm(hist[-1] - hist[-STUCK_WINDOW])) < STUCK_TRAVEL
            if noise > 0 and (mode == "always" or wedged):
                a[0] += rng.normal(0, noise * math.pi)      # push angle
                a[1] += rng.normal(0, noise)                # force
                a[2] += rng.normal(0, noise)                # spin torque
                a[0] = (a[0] + math.pi) % (2 * math.pi) - math.pi
                a[1] = float(np.clip(a[1], 0.0, 1.0))
                a[2] = float(np.clip(a[2], -1.0, 1.0))

            hist.append(env.state.obj.center.copy())
            _, _, term, trunc, info = env.step(
                a.astype(np.float32).reshape(env.action_space.shape))
            steps += 1
            done = term or trunc

        d = float(info.get("object_distance", env.state.distance_to_goal()))
        dists.append(d)
        succ.append(bool(info.get("is_success", False) or d < reach))
        travel = float(np.linalg.norm(hist[-1] - hist[max(0, len(hist) - STUCK_WINDOW)]))
        jam.append(travel < STUCK_TRAVEL)

    env.close()
    return (float(np.mean(succ) * 100), float(np.mean(dists)),
            float(np.median(dists)), float(np.min(dists)),
            float(np.mean(jam) * 100))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    p.add_argument("--ckpt", default="storage_local/checkpoints/bc_chunk_H1.pt")
    p.add_argument("--noises", default="0,0.02,0.05,0.1,0.2,0.4")
    p.add_argument("--modes", default="always,stuck")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    cfg = load_config(args.config)
    blob = torch.load(args.ckpt, map_location=dev)
    net = ChunkPolicy(27, blob.get("horizon", 1)).to(dev)
    net.load_state_dict(blob["state_dict"])
    net.eval()

    z = np.load(args.dataset)
    init_poses, goals = z["init_pose"], z["goal"]
    eval_eps = pick_eval_episodes(init_poses, args.episodes)
    logger.info(f"{len(eval_eps)} episodes with distinct start poses")

    rows, seen_zero = [], False
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        for ns in [n.strip() for n in args.noises.split(",") if n.strip()]:
            noise = float(ns)
            if noise == 0 and seen_zero:
                continue          # noise 0 behaves the same in every mode
            sr, mean_d, med_d, best_d, jam = run(
                net, cfg, init_poses, goals, eval_eps, noise, mode, dev)
            logger.info(f"[{mode} noise={noise:.2f}] SR={sr:.1f}% mean={mean_d:.4f} "
                        f"jammed={jam:.0f}%")
            rows.append((mode, noise, sr, mean_d, med_d, best_d, jam))
            if noise == 0:
                seen_zero = True

    print("\n" + "=" * 74)
    print(f"ACTION-NOISE SWEEP  ({len(eval_eps)} distinct start poses, {args.ckpt})")
    print("=" * 74)
    print(f"{'mode':<8} {'noise':>6} {'SR %':>7} {'mean dist':>11} {'median':>9} "
          f"{'best':>8} {'jammed %':>9}")
    print("-" * 74)
    for mode, noise, sr, mean_d, med_d, best_d, jam in rows:
        print(f"{mode:<8} {noise:>6.2f} {sr:>7.1f} {mean_d:>11.4f} {med_d:>9.4f} "
              f"{best_d:>8.4f} {jam:>9.0f}")
    print("\n'always' = noise on every step.  'stuck' = noise only after the load")
    print("has moved <10 mm for 100 steps, i.e. purely as a deadlock breaker.\n")


if __name__ == "__main__":
    main()
