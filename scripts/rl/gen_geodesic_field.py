"""Precompute a geodesic distance field over the free configuration space.

For every T pose (x, y, theta) on a grid, compute the number of BFS steps
(small translations/rotations through collision-free poses only) to the goal
set (= poses where the goal-tracked point is within reach_radius of the goal).
This is the "route distance around the walls", used by reward_mode: geodesic.

Usage:
    python gen_geodesic_field.py <config.yaml> --out storage_local/fields/x.npz
                                 [--dx 0.004] [--dth 3.0]

The field is tied to the config's wall geometry (walls, T shape, goal). Regenerate
after changing any of those.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ant_swarm import load_config  # noqa: E402
from ant_swarm.layout import Layout  # noqa: E402
from ant_swarm.tshape import TShape  # noqa: E402

UNREACHED = np.uint16(65535)


def free_mask(rects, walls_aabb, W, H, xs, ys, thetas, m=0.005):
    """free[theta, y, x]: vectorized SAT of the T's rects vs wall AABBs + world box."""
    ny, nx = len(ys), len(xs)
    X, Y = np.meshgrid(xs, ys)
    free = np.zeros((len(thetas), ny, nx), dtype=bool)
    wcs = [np.array([[a[0], a[2]], [a[1], a[2]], [a[1], a[3]], [a[0], a[3]]],
                    dtype=np.float64) for a in walls_aabb]
    for ti, th in enumerate(thetas):
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        ok = np.ones((ny, nx), dtype=bool)
        all_off = []
        for r in rects:
            hx, hy = r.half_size
            loc = r.center + np.array([[-hx, -hy], [-hx, hy], [hx, -hy], [hx, hy]])
            off = loc @ R.T
            all_off.append(off)
            axes = np.array([[1, 0], [0, 1], [c, s], [-s, c]])
            po = off @ axes.T
            pmin, pmax = po.min(0), po.max(0)
            pc = X[..., None] * axes[:, 0] + Y[..., None] * axes[:, 1]
            hit = np.zeros((ny, nx), dtype=bool)
            for wc in wcs:
                wp = wc @ axes.T
                wmin, wmax = wp.min(0), wp.max(0)
                hit |= ((pc + pmax >= wmin) & (pc + pmin <= wmax)).all(-1)
            ok &= ~hit
        allo = np.concatenate(all_off, 0)
        ok &= (X + allo[:, 0].min() >= m) & (X + allo[:, 0].max() <= W - m) \
            & (Y + allo[:, 1].min() >= m) & (Y + allo[:, 1].max() <= H - m)
        free[ti] = ok
    return free


def _neighbor_any(m):
    """6-neighborhood dilation; theta axis (0) wraps, x/y do not."""
    n = np.roll(m, 1, 0) | np.roll(m, -1, 0)
    n[:, 1:, :] |= m[:, :-1, :]
    n[:, :-1, :] |= m[:, 1:, :]
    n[:, :, 1:] |= m[:, :, :-1]
    n[:, :, :-1] |= m[:, :, 1:]
    return n


def _neighbor_min(d):
    """Min over the 6 neighbors (int32; theta wraps, x/y edges padded with max)."""
    big = np.int32(1 << 30)
    outs = [np.roll(d, 1, 0), np.roll(d, -1, 0)]
    for ax, sh in ((1, 1), (1, -1), (2, 1), (2, -1)):
        t = np.full_like(d, big)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if sh == 1:
            dst[ax], src[ax] = slice(1, None), slice(None, -1)
        else:
            dst[ax], src[ax] = slice(None, -1), slice(1, None)
        t[tuple(dst)] = d[tuple(src)]
        outs.append(t)
    return np.minimum.reduce(outs)


def main():
    argv = sys.argv[1:]
    cfg_path = next((a for a in argv if not a.startswith("--")), None)
    def opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default
    dx = float(opt("--dx", 0.004))
    dth = float(opt("--dth", 3.0))
    # Safety margin added to the walls for the BFS only: routes with less
    # clearance than this are excluded, so the field never guides the agent
    # into razor-thin channels (grid steps could also tunnel through them).
    inflate = float(opt("--inflate", 0.003))
    cfg = load_config(cfg_path)
    lay, tsh = Layout(cfg), TShape(cfg)
    W, H = lay.world_size
    walls_aabb = [(a[0] - inflate, a[1] + inflate, a[2] - inflate, a[3] + inflate)
                  for a in lay.walls_aabb]

    out = Path(opt("--out", "storage_local/fields/geodesic.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)

    xs = np.arange(0.02, W - 0.015, dx)
    ys = np.arange(0.02, H - 0.015, dx)
    thetas = np.radians(np.arange(-180.0, 180.0, dth))
    print(f"grid: {len(thetas)} x {len(ys)} x {len(xs)} = "
          f"{len(thetas) * len(ys) * len(xs) / 1e6:.1f}M cells", flush=True)

    t0 = time.time()
    free = free_mask(tsh.rects, walls_aabb, W, H, xs, ys, thetas,
                     m=0.005 + inflate)
    print(f"free space: {free.mean() * 100:.1f}% of cells "
          f"({time.time() - t0:.0f}s)", flush=True)

    # goal set: tracked point within reach radius (same test as env success)
    track = tsh.track_local_point(getattr(cfg.env, "goal_track", "center"))
    lx = float(track[0])
    gx, gy = float(lay.goal[0]), float(lay.goal[1])
    cos = np.cos(thetas)[:, None, None]
    sin = np.sin(thetas)[:, None, None]
    tx = xs[None, None, :] + lx * cos
    ty = ys[None, :, None] + lx * sin
    seeds = ((tx - gx) ** 2 + (ty - gy) ** 2 <= lay.reach_radius ** 2) & free
    print(f"goal seed cells: {seeds.sum()}", flush=True)
    assert seeds.any(), "goal set empty — check goal.pos / reach_radius vs grid"

    # BFS wavefront
    dist = np.full(free.shape, int(UNREACHED), dtype=np.int32)
    dist[seeds] = 0
    frontier = seeds
    d = 0
    while frontier.any():
        d += 1
        nxt = _neighbor_any(frontier) & free & (dist == int(UNREACHED))
        dist[nxt] = d
        frontier = nxt
    reachable = (dist < int(UNREACHED)) & free
    norm = int(dist[reachable].max())
    print(f"BFS done: max dist {norm} steps; reachable {reachable.sum() / max(free.sum(),1) * 100:.1f}% "
          f"of free space ({time.time() - t0:.0f}s)", flush=True)

    # fill blocked/unreached cells from neighbors so off-grid lookups near walls
    # still get a sensible value (gradients keep pointing along the route)
    for _ in range(40):
        unknown = dist == int(UNREACHED)
        if not unknown.any():
            break
        cand = _neighbor_min(dist)
        ok = unknown & (cand < int(UNREACHED))
        dist[ok] = cand[ok] + 1
    print(f"fill done: {(dist == int(UNREACHED)).sum()} cells left unreachable", flush=True)

    meta = dict(config=str(cfg_path or "config.yaml"),
                walls_x=list(map(float, cfg.walls.x_columns)),
                wall_len=float(cfg.walls.length),
                goal=[gx, gy], goal_track=str(getattr(cfg.env, "goal_track", "center")))
    np.savez_compressed(
        out, dist=np.minimum(dist, int(UNREACHED)).astype(np.uint16),
        norm=np.float64(norm),
        x0=xs[0], dx=np.float64(dx), y0=ys[0], dy=np.float64(dx),
        th0=np.float64(-180.0), dth=np.float64(dth),
        meta=json.dumps(meta))
    print(f"saved -> {out}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
