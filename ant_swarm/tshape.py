"""The movable T-shaped object: geometry, pose, transforms, and collision.

Built from the config namespace.  Collision is delegated to ``geometry`` (true
oriented-rectangle SAT) and tested against a ``Layout``'s wall AABBs.
"""
from __future__ import annotations

import numpy as np

from .geometry import LocalRect, obb_aabb_overlap, rotation_matrix


class TShape:
    """Rigid T-shape: three bars (stem, big cap, small cap) + a planar pose."""

    def __init__(self, cfg):
        s = float(cfg.scene_scale)
        self.scene_scale = s
        t = cfg.tshape

        self.stem_len = t.stem_len * s
        self.thickness = t.thickness * s
        self.cap_big_len = t.cap_big_len * s
        self.cap_small_len = t.cap_small_len * s

        ht = self.thickness / 2
        self.rects = [
            LocalRect(np.array([0.0, 0.0], dtype=np.float32),
                      np.array([self.stem_len / 2, ht], dtype=np.float32)),
            LocalRect(np.array([-self.stem_len / 2, 0.0], dtype=np.float32),
                      np.array([ht, self.cap_big_len / 2], dtype=np.float32)),
            LocalRect(np.array([self.stem_len / 2, 0.0], dtype=np.float32),
                      np.array([ht, self.cap_small_len / 2], dtype=np.float32)),
        ]

        self.center = np.zeros(2, dtype=np.float32)
        self.angle = 0.0
        self.vel = np.zeros(2, dtype=np.float32)
        self.ang_vel = 0.0

    # -- pose ----------------------------------------------------------
    def set_pose(self, center, angle):
        self.center = np.array(center, dtype=np.float32)
        self.angle = float(angle)
        self.vel = np.zeros(2, dtype=np.float32)
        self.ang_vel = 0.0
        return self

    def clone_at(self, center, angle):
        import copy
        return copy.copy(self).set_pose(center, angle)

    # -- transforms ----------------------------------------------------
    def rot(self):
        return rotation_matrix(self.angle)

    def world_to_local(self, p):
        return self.rot().T @ (p - self.center)

    def local_to_world(self, p):
        return self.center + self.rot() @ p

    # -- queries -------------------------------------------------------
    def contains(self, p, margin=0.0):
        q = self.world_to_local(p)
        for r in self.rects:
            d = np.abs(q - r.center) - (r.half_size + margin)
            if d[0] <= 0 and d[1] <= 0:
                return True
        return False

    def distance(self, p):
        q = self.world_to_local(p)
        best = 1e9
        for r in self.rects:
            d = np.abs(q - r.center) - r.half_size
            outside = np.maximum(d, 0.0)
            inside = min(max(d[0], d[1]), 0.0)
            best = min(best, float(np.linalg.norm(outside) + inside))
        return best

    def local_corners(self):
        corners = []
        for r in self.rects:
            hx, hy = r.half_size
            cx, cy = r.center
            corners += [[cx - hx, cy - hy], [cx - hx, cy + hy],
                        [cx + hx, cy - hy], [cx + hx, cy + hy]]
        return np.array(corners, dtype=np.float32)

    def world_corners(self):
        return self.center[None, :] + self.local_corners() @ self.rot().T

    # -- collision -----------------------------------------------------
    def overlaps_walls(self, layout) -> bool:
        """True oriented-rectangle collision against the layout's wall AABBs."""
        corners = self.world_corners()  # (12, 2): 4 per sub-rect
        for i in range(0, len(corners), 4):
            rc = corners[i:i + 4]
            for aabb in layout.walls_aabb:
                if obb_aabb_overlap(rc, aabb, self.angle):
                    return True
        return False


def sample_free_pose(tshape: TShape, layout, rng, *, x_range, angle_range,
                     margin=0.06, max_tries=500):
    """Sample a collision-free (center, angle) in ``x_range`` (scaled coords)."""
    s = layout.scene_scale
    W, H = layout.world_size
    x_lo, x_hi = x_range[0] * s, x_range[1] * s
    probe = tshape.clone_at(np.zeros(2), 0.0)
    for _ in range(max_tries):
        angle = float(rng.uniform(*angle_range))
        cx = float(rng.uniform(x_lo, x_hi))
        cy = float(rng.uniform(margin, H - margin))
        probe.set_pose([cx, cy], angle)
        c = probe.world_corners()
        if c[:, 0].min() < margin or c[:, 0].max() > W - margin:
            continue
        if c[:, 1].min() < margin or c[:, 1].max() > H - margin:
            continue
        if not probe.overlaps_walls(layout):
            return np.array([cx, cy], dtype=np.float32), angle
    return np.array([(x_lo + x_hi) / 2, H / 2], dtype=np.float32), 0.0


def make_attachment_offsets(tshape: TShape, n_ants: int, rng, ant_offsets=None) -> np.ndarray:
    """Local-frame attachment points for the ants on the T-shape.

    * explicit ``ant_offsets`` → used verbatim
    * ``n_ants == 2`` → one at each T-junction (stem ↔ cap)
    * otherwise → uniform random samples on the T perimeter
    """
    if ant_offsets is not None:
        return np.array(ant_offsets, dtype=np.float32)
    if n_ants == 2:
        return np.array([[-tshape.stem_len / 2, 0.0],
                         [ tshape.stem_len / 2, 0.0]], dtype=np.float32)

    rects = tshape.rects
    perims = [4.0 * (float(r.half_size[0]) + float(r.half_size[1])) for r in rects]
    total = sum(perims)
    probs = [p / total for p in perims]
    offsets = []
    for _ in range(n_ants):
        r_idx = int(rng.choice(len(rects), p=probs))
        rect = rects[r_idx]
        cx, cy = float(rect.center[0]), float(rect.center[1])
        hx, hy = float(rect.half_size[0]), float(rect.half_size[1])
        perim = 2 * (2 * hx + 2 * hy)
        t = rng.uniform(0, perim)
        if t < 2 * hx:
            x, y = cx - hx + t, cy - hy
        elif t < 2 * hx + 2 * hy:
            x, y = cx + hx, cy - hy + (t - 2 * hx)
        elif t < 4 * hx + 2 * hy:
            x, y = cx + hx - (t - 2 * hx - 2 * hy), cy + hy
        else:
            x, y = cx - hx, cy + hy - (t - 4 * hx - 2 * hy)
        offsets.append([x, y])
    return np.array(offsets, dtype=np.float32)
