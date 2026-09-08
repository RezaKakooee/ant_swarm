"""A/B test: waypoint conditioning vs global-goal conditioning for BC.

Motivation
----------
Handoff root cause #3: obs[4:6] is the global vector (goal - object).  When the
goal sits high in the goal room, that vector points UP while the load is still
inside the slit channel, so it pulls the T into the barrier before it has
cleared.

This script replaces obs[4:6] with the direction to a waypoint taken from the
pre-computed pose geodesic field (`storage_local/fields/pnas_gap015.npz`),
L descending BFS steps ahead of the current pose.  Inside the channel that
vector points ALONG the slit, so the upward pull disappears on its own.

The field was built for the fixed goal (1.2, 0.36) but the demos use random
goals in the right-hand room.  Every route must pass both slits, so the field
is used only up to the second wall column; past it the true goal vector is
restored.  Same fallback where the pose is not geodesically reachable.

Everything else is held fixed -- same network, same data, same torque head --
so any difference is attributable to the conditioning alone.

Usage
-----
    python scripts/il/train_waypoint_bc.py --cache storage_local/cache/replay_4000.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces.utils import flatten
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "il"))
from ant_swarm import AntSwarmEnv, load_config                      # noqa: E402
from ant_swarm.geodesic import PoseGeodesicField                    # noqa: E402
from test_torque_observability import collect                       # noqa: E402
from train_discrete_torque_bc import BCPolicy, train                # noqa: E402


# --------------------------------------------------------------------------- #
# waypoint lookup built once from the geodesic field
# --------------------------------------------------------------------------- #
class WaypointField:
    """Pose -> the grid pose `lookahead` descending BFS steps closer to goal."""

    def __init__(self, field: PoseGeodesicField, lookahead: int = 40):
        S = field.steps                      # (nth, ny, nx) int32 BFS step count
        R = field.reachable
        nth, ny, nx = S.shape
        n = nth * ny * nx
        flat = np.arange(n, dtype=np.int32).reshape(nth, ny, nx)

        # successor = any reachable 6-neighbour exactly one BFS step closer
        succ = flat.copy()
        found = np.zeros(S.shape, dtype=bool)
        target = S - 1
        movable = R & (S > 0)

        def take(nb_S, nb_R, nb_flat, valid=None):
            ok = (~found) & movable & nb_R & (nb_S == target)
            if valid is not None:
                ok &= valid
            succ[ok] = nb_flat[ok]
            found[ok] = True

        for shift in (-1, 1):                                   # theta wraps
            take(np.roll(S, shift, axis=0), np.roll(R, shift, axis=0),
                 np.roll(flat, shift, axis=0))
        for axis, size in ((1, ny), (2, nx)):                   # y and x clamp
            for shift in (-1, 1):
                valid = np.ones(S.shape, dtype=bool)
                sl = [slice(None)] * 3
                sl[axis] = slice(0, 1) if shift == 1 else slice(size - 1, size)
                valid[tuple(sl)] = False                        # no wrap-around
                take(np.roll(S, shift, axis=axis), np.roll(R, shift, axis=axis),
                     np.roll(flat, shift, axis=axis), valid)

        # walk the successor map `lookahead` times
        succ = succ.ravel()
        cur = np.arange(n, dtype=np.int32)
        for _ in range(lookahead):
            cur = succ[cur]

        ti, rem = np.divmod(cur.astype(np.int64), ny * nx)
        yi, xi = np.divmod(rem, nx)
        self.wp_x = (field.x0 + xi * field.dx).astype(np.float32)
        self.wp_y = (field.y0 + yi * field.dy).astype(np.float32)
        self.reachable_flat = R.ravel()
        self.field = field
        self.shape = (nth, ny, nx)
        logger.info(f"waypoint map built (lookahead={lookahead} BFS steps, "
                    f"{n/1e6:.1f}M cells)")

    def flat_index(self, x, y, theta):
        f = self.field
        nth, ny, nx = self.shape
        deg = (np.degrees(theta) - f.th0) % 360.0
        ti = np.rint(deg / f.dth).astype(np.int64) % nth
        yi = np.clip(np.rint((y - f.y0) / f.dy), 0, ny - 1).astype(np.int64)
        xi = np.clip(np.rint((x - f.x0) / f.dx), 0, nx - 1).astype(np.int64)
        return (ti * ny + yi) * nx + xi

    def lookup(self, x, y, theta):
        """Return (wp_x, wp_y, ok) for arrays of poses."""
        idx = self.flat_index(x, y, theta)
        return self.wp_x[idx], self.wp_y[idx], self.reachable_flat[idx]


# --------------------------------------------------------------------------- #
# observation rewrite
# --------------------------------------------------------------------------- #
def pose_from_obs(obs, W, H):
    """Recover (x, y, theta) -- obs[2:4] are centre/W,H and obs[6:8] are sin,cos."""
    return (obs[:, 2] * W, obs[:, 3] * H, np.arctan2(obs[:, 6], obs[:, 7]))


def unit_dir(obs):
    """Unit-normalise obs[4:6] so only the DIRECTION is conditioned on."""
    out = obs.copy()
    v = out[:, 4:6]
    out[:, 4:6] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return out


def waypoint_obs(obs, wpf, W, H, switch_x):
    """Copy of `obs` with the goal vector replaced by the waypoint vector.

    Past `switch_x`, or where the pose is unreachable, the original global goal
    vector is kept -- the field has one fixed goal but the demos do not.
    """
    x, y, th = pose_from_obs(obs, W, H)
    wx, wy, ok = wpf.lookup(x, y, th)
    use = ok & (x < switch_x)
    out = obs.copy()
    out[use, 4] = ((wx - x) / W).astype(np.float32)[use]
    out[use, 5] = ((wy - y) / H).astype(np.float32)[use]
    return out, use


def compare_directions(obs, wp, use, x, wall_x, margin=0.12):
    """Show that inside the channel the two conditionings disagree."""
    inside = use & (np.abs(x - wall_x) < margin)
    if not inside.any():
        logger.warning("no in-channel states to compare")
        return
    g, w = obs[inside, 4:6], wp[inside, 4:6]
    gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    wn = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
    cos = float((gn * wn).sum(axis=1).mean())
    logger.info(f"in-channel states: {int(inside.sum())}")
    logger.info(f"  goal vector     mean (dx, dy) = ({g[:,0].mean():+.4f}, {g[:,1].mean():+.4f})")
    logger.info(f"  waypoint vector mean (dx, dy) = ({w[:,0].mean():+.4f}, {w[:,1].mean():+.4f})")
    logger.info(f"  mean cosine between them = {cos:+.3f}  "
                f"(1.0 = identical, so lower means the fix actually changes something)")
    logger.info(f"  |dy| goal={np.abs(g[:,1]).mean():.4f} vs waypoint={np.abs(w[:,1]).mean():.4f} "
                f"(the upward pull)")


# --------------------------------------------------------------------------- #
# closed-loop eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(net, cfg, dataset_path, n_episodes, device, wpf, switch_x,
             normalize_dir=False, max_steps=500, seed=42):
    net.eval()
    dev = torch.device(device)
    z = np.load(dataset_path)
    init_poses, goals = z["init_pose"], z["goal"]
    env = AntSwarmEnv(config=cfg, seed=seed)
    W, H = env.layout.world_size
    reach = float(cfg.goal.reach_radius) * float(cfg.scene_scale)
    dists, succ = [], []

    from train_chunked_bc import pick_eval_episodes
    for ep in pick_eval_episodes(init_poses, n_episodes):
        env.reset(seed=int(seed) + int(ep), options={
            "init_pose": init_poses[ep], "goal": goals[ep]})

        info, done, steps = {}, False, 0
        while not done and steps < max_steps:
            o = flatten(env.observation_space,
                        env.obs_model.observe(env.state)).astype(np.float32)[None, :]
            if wpf is not None:
                o, _ = waypoint_obs(o, wpf, W, H, switch_x)
            if normalize_dir:
                o = unit_dir(o)
            a = net.act(torch.tensor(o, device=dev)).cpu().numpy()[0]
            _, _, term, trunc, info = env.step(
                a.astype(np.float32).reshape(env.action_space.shape))
            steps += 1
            done = term or trunc

        d = float(info.get("object_distance", env.state.distance_to_goal()))
        dists.append(d)
        succ.append(bool(info.get("is_success", False) or d < reach))

    env.close()
    return (float(np.mean(succ) * 100.0), float(np.mean(dists)),
            float(np.min(dists)), float(np.median(dists)))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/il/il_augmented_bc.yaml")
    p.add_argument("--dataset", default="storage_local/datasets/successes_v1/dataset.npz")
    p.add_argument("--cache", default="storage_local/cache/replay_4000.npz")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--field", default="storage_local/fields/pnas_gap015.npz")
    p.add_argument("--lookahead", type=int, default=40, help="BFS steps ahead")
    p.add_argument("--switch-x", type=float, default=-1.0,
                   help="use the true goal vector past this x (default: wall 2)")
    p.add_argument("--variants", default="goal,waypoint")
    p.add_argument("--normalize-dir", action="store_true",
                   help="unit-normalise obs[4:6] in BOTH variants, so the "
                        "goal vector's length is not a hidden extra input")
    p.add_argument("--torque-head", default="mse", choices=["mse", "bins"])
    p.add_argument("--bins", type=int, default=21)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--holdout", type=float, default=0.05)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dir", default="storage_local/checkpoints")
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")

    cfg = load_config(args.config)
    W = float(cfg.world.width) * float(cfg.scene_scale)
    H = float(cfg.world.height) * float(cfg.scene_scale)
    wall2_x = float(cfg.walls.x_columns[-1]) * float(cfg.scene_scale)
    switch_x = wall2_x if args.switch_x < 0 else args.switch_x
    logger.info(f"switch to true goal vector at x > {switch_x:.3f} m")

    cache = Path(args.cache)
    if cache.exists():
        logger.info(f"Loading replay cache {cache}")
        z = np.load(cache)
        data = {k: z[k] for k in z.files}
    else:
        data = collect(cfg, args.dataset, args.episodes)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
    obs, act, epid = data["obs"], data["act"], data["epid"]
    from repair_replay_goals import require_correct_goals
    with np.load(args.dataset) as demonstrations:
        require_correct_goals(obs, epid, demonstrations["goal"], cfg)
    logger.info(f"{len(obs)} transitions from {len(np.unique(epid))} episodes")

    field = PoseGeodesicField(args.field)
    field.validate_config(cfg, check_goal=False)
    wpf = WaypointField(field, args.lookahead)

    wp_obs, used = waypoint_obs(obs, wpf, W, H, switch_x)
    x_all = obs[:, 2] * W
    logger.info(f"waypoint substituted on {100*used.mean():.1f}% of steps")
    dstep = np.hypot(wp_obs[used, 4] * W, wp_obs[used, 5] * H)
    logger.info(f"mean waypoint distance = {dstep.mean():.3f} m")
    compare_directions(obs, wp_obs, used, x_all, wall2_x)

    uniq = np.unique(epid)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    held = set(uniq[:max(1, int(len(uniq) * args.holdout))].tolist())
    is_held = np.array([e in held for e in epid])

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        X = wp_obs if variant == "waypoint" else obs
        if args.normalize_dir:
            X = unit_dir(X)
        t0 = time.time()
        net = train(X[~is_held], act[~is_held], args.torque_head, args.bins,
                    args.epochs, args.batch_size, args.lr, device)
        tag = f"{variant}_{args.torque_head}_L{args.lookahead}"
        tag += "_norm" if args.normalize_dir else ""
        ckpt = save_dir / f"bc_{tag}.pt"
        torch.save({"state_dict": net.state_dict(), "variant": variant,
                    "torque_head": args.torque_head, "lookahead": args.lookahead,
                    "switch_x": switch_x}, ckpt)
        logger.info(f"[{variant}] closed-loop eval on {args.eval_episodes} episodes ...")
        sr, mean_d, best_d, med_d = evaluate(
            net, cfg, args.dataset, args.eval_episodes, device,
            wpf if variant == "waypoint" else None, switch_x,
            normalize_dir=args.normalize_dir)
        logger.info(f"[{variant}] SR={sr:.1f}% mean={mean_d:.4f} median={med_d:.4f} "
                    f"best={best_d:.4f} ({time.time()-t0:.0f}s) -> {ckpt}")
        results.append((variant, sr, mean_d, med_d, best_d))

    print("\n" + "=" * 72)
    print(f"CONDITIONING A/B  (torque head={args.torque_head}, "
          f"lookahead={args.lookahead}, {args.epochs} epochs)")
    print("=" * 72)
    print(f"{'variant':<10} {'SR %':>7} {'mean dist':>11} {'median dist':>12} {'best dist':>11}")
    print("-" * 72)
    for variant, sr, mean_d, med_d, best_d in results:
        print(f"{variant:<10} {sr:>7.1f} {mean_d:>11.4f} {med_d:>12.4f} {best_d:>11.4f}")
    print()


if __name__ == "__main__":
    main()
