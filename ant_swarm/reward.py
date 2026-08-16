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
    make real progress are rewarded. This is a pretraining signal; sparse
    fine-tuning restores the exact terminal objective.
    Config: ``env.geodesic_field`` = .npz path (relative to project root).
    The field is tied to ONE wall geometry — safe with the reverse curriculum
    (fixed walls), NOT with the gap curriculum.

``reached`` = the goal-tracking point is within the goal's reach radius.
"""
from __future__ import annotations

from .geodesic import PoseGeodesicField


class RewardModel:
    def __init__(self, cfg):
        self.mode = getattr(cfg.env, "reward_mode", "shaped")
        self.progress_coef = cfg.env.reward_progress_coef
        self.success = cfg.env.reward_success
        self._prev_phi = 0.0
        if self.mode == "geodesic":
            self.geo = PoseGeodesicField(cfg.env.geodesic_field)
            self.geo.validate_config(cfg)
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
