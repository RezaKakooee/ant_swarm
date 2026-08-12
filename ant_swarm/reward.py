"""Reward component.

Three modes (config ``env.reward_mode``):

  * ``shaped`` (default): dense distance shaping + sparse success bonus::
        reward = progress_coef * (prev_dist - dist) + (success if reached)
  * ``sparse``: only the success bonus::
        reward = success if reached else 0
  * ``geodesic``: dense shaping on a precomputed geodesic distance field over
    the free configuration space (x, y, theta)::
        reward = geodesic_coef * (phi_prev - phi) + (success if reached)
    where phi = normalised BFS distance-to-goal along collision-free motions
    (see ``gen_geodesic_field.py``). Unlike Euclidean distance this decreases
    monotonically along the true solution path, so rotations and detours that
    make real progress are rewarded. Potential-based -> policy-invariant.
    Config: ``env.geodesic_field`` = .npz path (relative to project root).
    The field is tied to ONE wall geometry — safe with the reverse curriculum
    (fixed walls), NOT with the gap curriculum.

``reached`` = the goal-tracking point is within the goal's reach radius.
"""
from __future__ import annotations

import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


class _GeoField:
    """Nearest-cell lookup into a precomputed (theta, y, x) distance field."""

    def __init__(self, path: str | Path):
        import numpy as np
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        z = np.load(p)
        d = z["dist"].astype(np.float32)
        self.norm = float(z["norm"])
        d[d >= 65500.0] = self.norm * 1.2          # unreachable → worse than any real pose
        self.d = d / self.norm                     # phi in ~[0, 1.2]
        self.x0, self.dx = float(z["x0"]), float(z["dx"])
        self.y0, self.dy = float(z["y0"]), float(z["dy"])
        self.th0, self.dth = float(z["th0"]), float(z["dth"])  # degrees
        self.nth, self.ny, self.nx = self.d.shape

    def phi(self, x: float, y: float, angle: float) -> float:
        deg = math.degrees(angle)
        deg = (deg - self.th0) % 360.0
        i = int(round(deg / self.dth)) % self.nth
        j = min(max(int(round((y - self.y0) / self.dy)), 0), self.ny - 1)
        k = min(max(int(round((x - self.x0) / self.dx)), 0), self.nx - 1)
        return float(self.d[i, j, k])


class RewardModel:
    def __init__(self, cfg):
        self.mode = getattr(cfg.env, "reward_mode", "shaped")
        self.progress_coef = cfg.env.reward_progress_coef
        self.success = cfg.env.reward_success
        self._prev_phi = 0.0
        if self.mode == "geodesic":
            self.geo = _GeoField(cfg.env.geodesic_field)
            self.geo_coef = float(getattr(cfg.env, "reward_geodesic_coef", 1.0))

    # ------------------------------------------------------------------
    def _phi(self, state) -> float:
        c = state.obj.center
        return self.geo.phi(float(c[0]), float(c[1]), float(state.obj.angle))

    def reset(self, state) -> None:
        """Re-anchor the potential at episode start (env calls this on reset)."""
        if self.mode == "geodesic":
            self._prev_phi = self._phi(state)

    def compute(self, dist: float, prev_dist: float, reach_radius: float, state=None):
        """Return ``(reward, reached)`` for one transition."""
        reached = dist < reach_radius
        bonus = self.success if reached else 0.0
        if self.mode == "sparse":
            return bonus, reached
        if self.mode == "geodesic":
            phi = self._phi(state)
            reward = self.geo_coef * (self._prev_phi - phi) + bonus
            self._prev_phi = phi
            return reward, reached
        return self.progress_coef * (prev_dist - dist) + bonus, reached
