"""Dynamic world state + physics integration.

``SwarmState`` owns the mutable simulation state — the posed T-shape, the ant
world positions, and the step counter — and advances it under an applied
(force, torque) wrench with wall + boundary collision response.
"""
from __future__ import annotations

import numpy as np


class SwarmState:
    """Mutable physics state of one episode."""

    def __init__(self, cfg, layout, tshape_template, attachment_offsets):
        s = float(cfg.scene_scale)
        self.layout = layout
        self.template = tshape_template
        self.attachment_offsets = np.asarray(attachment_offsets, dtype=np.float32)
        self.n_ants = len(self.attachment_offsets)

        ph = cfg.physics
        self.object_mass = ph.object_mass
        self.object_inertia = ph.object_inertia * s * s
        self.linear_friction = ph.linear_friction
        self.angular_friction = ph.angular_friction
        self.substeps = int(ph.substeps)
        self.boundary_margin = ph.boundary_margin * s
        self.rest_wall = ph.restitution_wall
        self.rest_bound = ph.restitution_bound

        self.obj = None          # posed TShape for the current episode
        self.ants = None         # (n_ants, 2) world positions
        self.step_count = 0

    # ------------------------------------------------------------------
    def reset(self, center, angle):
        self.obj = self.template.clone_at(center, angle)
        self.step_count = 0
        self._update_ants()
        return self

    def _update_ants(self):
        self.ants = np.array(
            [self.obj.local_to_world(self.attachment_offsets[i]) for i in range(self.n_ants)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def attachment_world(self, i):
        return self.obj.local_to_world(self.attachment_offsets[i])

    @property
    def object_center(self):
        return self.obj.center

    @property
    def object_angle(self):
        return self.obj.angle

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.obj.center - self.layout.goal))

    # ------------------------------------------------------------------
    def integrate(self, force, torque):
        """Advance the T-shape one env step under a wrench, resolving collisions."""
        obj, lay = self.obj, self.layout
        obj.vel = self.linear_friction * obj.vel + force / self.object_mass
        obj.ang_vel = self.angular_friction * obj.ang_vel + torque / self.object_inertia

        n_sub = self.substeps
        rw = self.rest_wall
        dp = obj.vel / n_sub
        da = obj.ang_vel / n_sub
        for _ in range(n_sub):
            prev_c = obj.center.copy()
            prev_a = obj.angle
            obj.center = prev_c + dp
            obj.angle = prev_a + da
            if obj.overlaps_walls(lay):
                obj.center = np.array([prev_c[0] + dp[0], prev_c[1]], dtype=np.float32)
                if obj.overlaps_walls(lay):
                    obj.center = np.array([prev_c[0], prev_c[1] + dp[1]], dtype=np.float32)
                    if obj.overlaps_walls(lay):
                        obj.center = prev_c
                        obj.angle = prev_a
                        obj.vel *= rw
                        obj.ang_vel *= rw
                        break
                    else:
                        obj.vel[0] *= rw
                else:
                    obj.vel[1] *= rw
            if obj.overlaps_walls(lay):
                obj.angle = prev_a
                obj.ang_vel *= rw

        W, H = lay.world_size
        corners = obj.world_corners()
        shift = np.zeros(2, dtype=np.float32)
        m = self.boundary_margin
        rb = self.rest_bound
        if corners[:, 0].min() < m:
            shift[0] += m - corners[:, 0].min(); obj.vel[0] *= rb
        if corners[:, 0].max() > W - m:
            shift[0] -= corners[:, 0].max() - (W - m); obj.vel[0] *= rb
        if corners[:, 1].min() < m:
            shift[1] += m - corners[:, 1].min(); obj.vel[1] *= rb
        if corners[:, 1].max() > H - m:
            shift[1] -= corners[:, 1].max() - (H - m); obj.vel[1] *= rb
        obj.center += shift

        self._update_ants()
