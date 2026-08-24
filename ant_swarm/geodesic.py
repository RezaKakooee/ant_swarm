"""Pose-space geodesic field lookup and collision-free path extraction.

The field is produced by :mod:`scripts.rl.gen_geodesic_field`. Its axes are
``(theta, y, x)`` and each cell contains the 6-connected BFS distance to the
goal set. Following strictly decreasing cells therefore yields a valid path
in ``(x, y, theta)`` rather than a misleading straight line through walls.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
UNREACHED = 65500


class PoseGeodesicField:
    """Load a pose field, query its potential, and extract descending paths."""

    def __init__(self, path: str | Path):
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        z = np.load(p)
        raw = z["dist"].astype(np.float32)
        self.norm = float(z["norm"])
        self.has_reachable_mask = "reachable" in z
        self.reachable = (z["reachable"].astype(bool)
                          if self.has_reachable_mask else raw < UNREACHED)
        self.steps = z["dist"].astype(np.int32)
        raw[raw >= UNREACHED] = self.norm * 1.2
        self.d = raw / self.norm
        self.x0, self.dx = float(z["x0"]), float(z["dx"])
        self.y0, self.dy = float(z["y0"]), float(z["dy"])
        self.th0, self.dth = float(z["th0"]), float(z["dth"])
        self.nth, self.ny, self.nx = self.d.shape
        self.meta = {}
        if "meta" in z:
            try:
                self.meta = json.loads(str(z["meta"].item()))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid geodesic field metadata in {p}") from exc

    def validate_config(self, cfg, check_goal: bool = True) -> None:
        """Reject a field generated for different task geometry.

        ``check_goal=False`` when several fields are loaded, one per goal:
        the geometry must match, but each field has its own goal.
        """
        if not self.meta:
            return
        s = float(cfg.scene_scale)
        expected = {
            "scene_scale": s,
            "world": [float(cfg.world.width), float(cfg.world.height)],
            "walls_x": list(map(float, cfg.walls.x_columns)),
            "wall_len": float(cfg.walls.length),
            "wall_thickness": float(cfg.walls.thickness),
            "goal": [float(v) * s for v in cfg.goal.pos],
            "reach_radius": float(cfg.goal.reach_radius) * s,
            "goal_track": str(getattr(cfg.env, "goal_track", "center")),
        }
        tshape = {
            "stem_len": float(cfg.tshape.stem_len),
            "cap_big_len": float(cfg.tshape.cap_big_len),
            "cap_small_len": float(cfg.tshape.cap_small_len),
            "thickness": float(cfg.tshape.thickness),
        }
        mismatches = []
        for key, wanted in expected.items():
            if key not in self.meta:
                continue
            if not check_goal and key == "goal":
                continue
            actual = self.meta[key]
            if isinstance(wanted, str):
                same = str(actual) == wanted
            else:
                same = np.allclose(np.asarray(actual, dtype=float),
                                   np.asarray(wanted, dtype=float),
                                   rtol=0.0, atol=1e-6)
            if not same:
                mismatches.append(key)
        if "tshape" in self.meta:
            for key, wanted in tshape.items():
                if key in self.meta["tshape"] and not math.isclose(
                        float(self.meta["tshape"][key]), wanted,
                        rel_tol=0.0, abs_tol=1e-6):
                    mismatches.append(f"tshape.{key}")
        if mismatches:
            raise ValueError(
                "geodesic field does not match config: " + ", ".join(mismatches))


    def index(self, x: float, y: float, angle: float) -> tuple[int, int, int]:
        deg = (math.degrees(angle) - self.th0) % 360.0
        ti = int(round(deg / self.dth)) % self.nth
        yi = min(max(int(round((y - self.y0) / self.dy)), 0), self.ny - 1)
        xi = min(max(int(round((x - self.x0) / self.dx)), 0), self.nx - 1)
        return ti, yi, xi

    def pose(self, index: tuple[int, int, int]) -> np.ndarray:
        ti, yi, xi = index
        angle = math.radians(self.th0 + ti * self.dth)
        angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
        return np.array([self.x0 + xi * self.dx,
                         self.y0 + yi * self.dy,
                         angle], dtype=np.float32)

    def phi(self, x: float, y: float, angle: float) -> float:
        return float(self.d[self.index(x, y, angle)])

    def _neighbors(self, index: tuple[int, int, int]):
        ti, yi, xi = index
        yield ((ti - 1) % self.nth, yi, xi)
        yield ((ti + 1) % self.nth, yi, xi)
        if yi > 0:
            yield (ti, yi - 1, xi)
        if yi + 1 < self.ny:
            yield (ti, yi + 1, xi)
        if xi > 0:
            yield (ti, yi, xi - 1)
        if xi + 1 < self.nx:
            yield (ti, yi, xi + 1)

    def path_from(self, pose) -> np.ndarray:
        """Return a start-to-goal path of valid grid poses."""
        if not self.has_reachable_mask:
            raise ValueError(
                "geodesic field has no reachable mask; regenerate it with "
                "scripts/rl/gen_geodesic_field.py before extracting a path")
        current = self.index(float(pose[0]), float(pose[1]), float(pose[2]))
        if not self.reachable[current]:
            raise ValueError(f"start pose {tuple(map(float, pose))} is not geodesically reachable")
        path = [self.pose(current)]
        remaining = int(self.steps[current])
        while remaining > 0:
            choices = [q for q in self._neighbors(current)
                       if self.reachable[q] and int(self.steps[q]) == remaining - 1]
            if not choices:
                raise RuntimeError("geodesic field has no descending neighbor")
            current = min(choices, key=lambda q: int(self.steps[q]))
            remaining = int(self.steps[current])
            path.append(self.pose(current))
        return np.asarray(path, dtype=np.float32)

    def farthest_pose(self, *, x_range, angle_range=(-math.pi, math.pi)) -> np.ndarray:
        """Choose the farthest reachable grid pose inside the spawn ranges."""
        xs = self.x0 + np.arange(self.nx) * self.dx
        angles = np.radians(self.th0 + np.arange(self.nth) * self.dth)
        angle_ok = np.ones(self.nth, dtype=bool)
        lo, hi = float(angle_range[0]), float(angle_range[1])
        if hi - lo < 2 * math.pi - 1e-5:
            wrapped = (angles + math.pi) % (2 * math.pi) - math.pi
            angle_ok = (wrapped >= lo) & (wrapped <= hi)
        mask = self.reachable.copy()
        mask &= angle_ok[:, None, None]
        mask &= ((xs >= float(x_range[0])) & (xs <= float(x_range[1])))[None, None, :]
        if not mask.any():
            raise ValueError("no geodesically reachable pose lies in spawn ranges")
        score = np.where(mask, self.steps, -1)
        return self.pose(tuple(np.unravel_index(int(np.argmax(score)), score.shape)))


def curriculum_anchors(field: PoseGeodesicField, start_pose, stages: int,
                       near_goal_fraction: float = 0.05) -> np.ndarray:
    """Return near-goal-to-full-start anchors sampled from one valid path."""
    if stages < 2:
        raise ValueError("pose-path curriculum needs at least two stages")
    path = field.path_from(start_pose)
    near = min(max(float(near_goal_fraction), 0.0), 0.95)
    first = max(0, min(len(path) - 2,
                       int(round((1.0 - near) * (len(path) - 1)))))
    count = min(stages, first + 1)
    indices = np.rint(np.linspace(first, 0, count)).astype(int)
    indices = np.asarray(list(dict.fromkeys(indices.tolist())), dtype=int)
    if indices[-1] != 0:
        indices = np.append(indices, 0)
    return path[indices]
