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

  * ``geodesic_exit``: two-leg reward for random-goal training. Leg 1: one
    SHARED geodesic field toward a fixed exit point past the second wall —
    identical for every goal, so the hard maneuver is learned once. Leg 2:
    once the load is fully past the second wall, plain distance shaping toward
    the actual (random) goal. Both legs are potential-based; the goal only
    matters in the easy open-room leg.
    Config: ``env.geodesic_field`` (exit field), ``env.reward_dist_coef``.

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
        self.fields = None
        if self.mode == "geodesic_exit":
            # leg 1: shared field to a fixed exit point (its own "goal")
            self.geo = PoseGeodesicField(cfg.env.geodesic_field)
            self.geo.validate_config(cfg, check_goal=False)
            self.geo_coef = float(getattr(cfg.env, "reward_geodesic_coef", 1.0))
            self.dist_coef = float(getattr(cfg.env, "reward_dist_coef", 0.5))
            # leg 2 starts when every corner of the load is past this x
            s = float(cfg.scene_scale)
            self._switch_x = (max(cfg.walls.x_columns) + cfg.walls.thickness) * s
        if self.mode == "geodesic":
            paths = getattr(cfg.env, "geodesic_fields", None)
            if paths:
                # random-goal training: one precomputed field per goal position
                self.fields = {}
                for path in paths:
                    field = PoseGeodesicField(path)
                    field.validate_config(cfg, check_goal=False)
                    self.fields[self._key(field.meta["goal"])] = field
                self.geo = next(iter(self.fields.values()))
            else:
                self.geo = PoseGeodesicField(cfg.env.geodesic_field)
                self.geo.validate_config(cfg)
            self.geo_coef = float(getattr(cfg.env, "reward_geodesic_coef", 1.0))

    @staticmethod
    def _key(goal):
        return tuple(round(float(v), 4) for v in goal)

    def set_goal(self, goal) -> None:
        """Switch to the field built for this goal (random-goal training)."""
        if self.mode != "geodesic" or not self.fields:
            return
        key = self._key(goal)
        field = self.fields.get(key)
        if field is None:
            raise KeyError(f"no geodesic field for goal {key}; "
                           f"have {sorted(self.fields)}")
        self.geo = field

    # ------------------------------------------------------------------
    def _past_walls(self, state) -> bool:
        return bool(state.obj.world_corners()[:, 0].min() > self._switch_x)

    def _potential(self, state, dist: float) -> float:
        """Piecewise potential for ``geodesic_exit``. The small jump at the
        switch is potential-based too, so the policy objective is unchanged."""
        if self._past_walls(state):
            return -self.dist_coef * dist          # leg 2: pull to the real goal
        return -self.geo_coef * self._phi(state)   # leg 1: thread the slits

    def _phi(self, state) -> float:
        c = state.obj.center
        return self.geo.phi(float(c[0]), float(c[1]), float(state.obj.angle))

    def reset(self, state) -> None:
        """Re-anchor the potential at episode start (env calls this on reset)."""
        if self.mode == "geodesic":
            self._prev_phi = self._phi(state)
        elif self.mode == "geodesic_exit":
            self._prev_phi = self._potential(state, state.distance_to_goal())

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
        if self.mode == "geodesic_exit":
            pot = self._potential(state, dist)
            reward = (pot - self._prev_phi) + bonus
            self._prev_phi = pot
            return reward, reached
        return self.progress_coef * (prev_dist - dist) + bonus, reached
