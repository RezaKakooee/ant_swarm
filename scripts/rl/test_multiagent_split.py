"""Option C: is multi-ant hard because of physics, or only because of learning?

Take the solved single-ant policy. It outputs a wrench: a push force plus a
direct spin torque. With n >= 2 there is no spin action -- torque can only come
from pushing at different attachment points. So: reproduce the single-ant wrench
with n ants by least squares, and see whether the maze is still solved.

    sum_i f_i                     = F_desired
    sum_i (arm_i x f_i)           = T_desired

3 equations, 2n unknowns. For n >= 2 the minimum-norm solution is taken. Each
ant's force is capped at push_strength, so if the request is infeasible the whole
wrench is scaled down and that step is counted.

If this solves the maze, the physics is fine and multi-agent is purely a
coordination-learning problem. If it does not, n ants cannot reproduce what the
single ant does and the task itself changes.

No learning happens here. It is a feasibility probe.

    python scripts/rl/test_multiagent_split.py --ants 2 --episodes 50
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "il"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
from ant_swarm import AntSwarmEnv, load_config      # noqa: E402
from train_chunked_bc import ChunkPolicy            # noqa: E402


def single_ant_obs(multi_obs: np.ndarray, n_ants: int) -> np.ndarray:
    """Rebuild the n=1 observation from any ant's row.

    Two things are n-dependent and must be undone, or the single-ant policy is
    fed an observation it was never trained on:

      obs[0:2]   attachment offset. The single ant sits at the frame origin.
      obs[9:11]  linear velocity, normalised by
                 (push_strength * n_ants / mass / damping) in
                 ObservationModel. The scale is proportional to n_ants, so
                 multiplying by n_ants restores the n=1 normalisation.

    Everything else (object pose, goal vector, orientation, angular velocity,
    barrier distances) is shared across ants and needs no change.
    """
    obs = np.asarray(multi_obs, dtype=np.float32)[0].copy()
    obs[0:2] = 0.0
    obs[9:11] = np.clip(obs[9:11] * float(n_ants), -1.0, 1.0)
    return obs


def split_wrench(force_xy, torque, arms, max_force):
    """Least-norm per-ant forces reproducing (force_xy, torque).

    Returns (forces (n,2), scaled) where `scaled` is True if the request had to
    be shrunk to respect the per-ant force cap.
    """
    n = len(arms)
    A = np.zeros((3, 2 * n), dtype=np.float64)
    A[0, 0::2] = 1.0                       # sum fx
    A[1, 1::2] = 1.0                       # sum fy
    A[2, 0::2] = -arms[:, 1]               # arm_x*fy - arm_y*fx
    A[2, 1::2] = arms[:, 0]
    b = np.array([force_xy[0], force_xy[1], torque], dtype=np.float64)

    f = np.linalg.lstsq(A, b, rcond=None)[0].reshape(n, 2)
    mags = np.linalg.norm(f, axis=1)
    scaled = False
    worst = mags.max() / max_force if max_force > 0 else 0.0
    if worst > 1.0:
        f = f / worst                      # keep direction, drop magnitude
        scaled = True
    return f, scaled


def rollout(env, net, cfg, max_steps, dev, frames=None, every=2):
    ph = cfg.physics
    s = float(cfg.scene_scale)
    push = float(ph.push_strength) * s
    spin_cfg = ph.get("spin_strength", None)
    spin_strength = (float(spin_cfg) * s if spin_cfg is not None
                     else push * (float(cfg.tshape.stem_len) * s) / 2)
    offsets = env.attachment_offsets
    n = len(offsets)

    obs = env.obs_model.observe(env.state)
    info, done, steps, n_scaled = {}, False, 0, 0
    if frames is not None:
        frames.append(env.render())
    while not done and steps < max_steps:
        with torch.no_grad():
            a = net(torch.tensor(single_ant_obs(obs, n)[None, :],
                                 device=dev))[0, 0].cpu().numpy()
        angle, mag, spin = float(a[0]), float(np.clip(a[1], 0, 1)), float(np.clip(a[2], -1, 1))
        F = push * mag * np.array([math.cos(angle), math.sin(angle)])
        T = spin * spin_strength

        if env.action_model.single_spin:
            # n=1 keeps the direct spin action: pass the policy through
            action = np.array([[angle, mag, spin]], dtype=np.float32)
        else:
            arms = offsets @ env.state.obj.rot().T      # local -> world arms
            f, scaled = split_wrench(F, T, arms, push)
            n_scaled += int(scaled)
            action = np.zeros((n, 2), dtype=np.float32)
            action[:, 0] = np.arctan2(f[:, 1], f[:, 0])
            action[:, 1] = np.clip(np.linalg.norm(f, axis=1) / push, 0.0, 1.0)
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc
        if frames is not None and (steps % every == 0 or done):
            frames.append(env.render())

    d = float(info.get("object_distance", env.state.distance_to_goal()))
    return d, bool(info.get("is_success", False)), steps, n_scaled / max(steps, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--ckpt",
                   default="storage_local/ant__20260905_1322__240805__train_goal_bc/bc_goal_fixed.pt")
    p.add_argument("--ants", default="1,2,3,4", help="comma list of n to try")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--render", type=int, default=0,
                   help="render this many episodes per ant count as GIFs")
    p.add_argument("--out", default=None, help="default: storage_local/<run id>")
    args = p.parse_args()

    dev = torch.device(args.device)
    blob = torch.load(args.ckpt, map_location=dev, weights_only=True)
    if blob.get("goal_observation_version") != 2:
        raise SystemExit("checkpoint predates the goal fix; retrain it first")
    net = ChunkPolicy(27, int(blob.get("horizon", 1))).to(dev)
    net.load_state_dict(blob["state_dict"])
    net.eval()

    out = None
    if args.render:
        from ant_swarm.run_id import build_run_id
        out = Path(args.out) if args.out else (
            Path('storage_local') / build_run_id('multiagent_split'))
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f'rendering into {out}')

    rows = []
    for ns in [x.strip() for x in args.ants.split(",") if x.strip()]:
        n = int(ns)
        cfg = load_config(args.config).copy()
        cfg.ants.n = n
        env = AntSwarmEnv(config=cfg, seed=args.seed)
        dists, succ, scaled = [], [], []
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            frames = [] if (out is not None and ep < args.render) else None
            d, ok, steps, frac = rollout(env, net, cfg, args.max_steps, dev, frames)
            dists.append(d); succ.append(ok); scaled.append(frac)
            if frames:
                from PIL import Image
                imgs = [Image.fromarray(f) for f in frames]
                tag = 'success' if ok else 'failure'
                name = out / f'ants{n}_ep{ep}_{tag}.gif'
                imgs[0].save(name, save_all=True, append_images=imgs[1:],
                             loop=0, duration=80, optimize=True)
                logger.info(f'  {name.name}  dist={d:.4f}  steps={steps}')
        env.close()
        sr = float(np.mean(succ) * 100)
        rows.append((n, sr, float(np.mean(dists)), float(np.median(dists)),
                     float(np.mean(scaled) * 100)))
        logger.info(f"n={n}: SR={sr:.1f}%  mean={np.mean(dists):.4f}  "
                    f"force-capped steps={np.mean(scaled)*100:.1f}%")

    print("\n" + "=" * 68)
    print(f"WRENCH-SPLIT PROBE — single-ant policy driving n ants "
          f"({args.episodes} episodes, seed {args.seed})")
    print("=" * 68)
    print(f"{'ants':>5} {'SR %':>8} {'mean dist':>11} {'median':>10} {'force-capped %':>16}")
    print("-" * 68)
    for n, sr, md, med, sc in rows:
        print(f"{n:>5} {sr:>8.1f} {md:>11.4f} {med:>10.4f} {sc:>16.1f}")
    print("\nn=1 is the reference (the policy's own setting: it has a spin action).")
    print("For n>=2 torque comes only from the geometry of the pushes.")
    print("High SR => physics is fine, multi-agent is a coordination-LEARNING problem.")
    print("Low SR  => n ants cannot reproduce the single-ant wrench; the task changes.\n")


if __name__ == "__main__":
    main()
