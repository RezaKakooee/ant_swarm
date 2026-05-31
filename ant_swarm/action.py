"""Action component: per-ant radial force → net wrench on the T-shape.

Action layout (per ant): ``[angle ∈ [-π, π], magnitude ∈ [0, 1]]``.
Each ant applies ``push_strength * magnitude * [cos, sin]`` at its attachment
point; the env sums these into a (force, torque) wrench.
"""
from __future__ import annotations

import math

import numpy as np

from ._gym import spaces


class ActionModel:
    def __init__(self, cfg):
        s = float(cfg.scene_scale)
        self.n_ants = int(cfg.ants.n)
        self.push_strength = cfg.physics.push_strength * s

    def space(self) -> spaces.Box:
        n = self.n_ants
        return spaces.Box(
            low=np.array([[-math.pi, 0.0]] * n, dtype=np.float32),
            high=np.array([[math.pi, 1.0]] * n, dtype=np.float32),
            dtype=np.float32,
        )

    def to_wrench(self, actions, state):
        """Sum per-ant radial forces into a net (force, torque) about the centre."""
        actions = np.asarray(actions, dtype=np.float32)  # (n_ants, 2)
        total_force = np.zeros(2, dtype=np.float32)
        total_torque = 0.0
        for i in range(self.n_ants):
            angle = float(actions[i, 0])
            magnitude = float(np.clip(actions[i, 1], 0.0, 1.0))
            if magnitude < 1e-8:
                continue
            move_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            force = self.push_strength * magnitude * move_dir
            arm = state.attachment_world(i) - state.object_center
            total_force += force
            total_torque += arm[0] * force[1] - arm[1] * force[0]
        return total_force, total_torque
