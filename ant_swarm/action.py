"""Action component: agent action(s) → motion command for the T-shape.

Two motion modes (set ``motion.mode`` in config):

KINEMATIC (simple, default):
    Single continuous action ``[direction ∈ [-π, π], rotation ∈ [-1, 1]]`` for
    the whole T. Each step the T translates ``step_len`` along ``direction`` and
    rotates ``rot_step * rotation`` — no forces, mass, or momentum.

DYNAMIC (physics):
    Multi-agent: per-ant ``[angle, magnitude]`` → forces; rotation from
    differential forces across ants.
    Single-agent (n==1): ``[angle, magnitude, spin]`` (spin → torque, since a
    central point force can't rotate the body).
"""
from __future__ import annotations

import math

import numpy as np

from ._gym import spaces


class ActionModel:
    def __init__(self, cfg):
        s = float(cfg.scene_scale)
        motion = getattr(cfg, "motion", None)
        self.mode = getattr(motion, "mode", "dynamic") if motion else "dynamic"
        self.n_ants = int(cfg.ants.n)

        if self.mode == "kinematic":
            self.step_len = motion.step_len * s
            self.rot_step = motion.rot_step
        else:
            self.push_strength = cfg.physics.push_strength * s
            self.single_spin = (self.n_ants == 1) and bool(getattr(cfg.ants, "single_agent_spin", True))
            spin_cfg = getattr(cfg.physics, "spin_strength", None)
            self.spin_strength = (spin_cfg * s if spin_cfg is not None
                                  else self.push_strength * (cfg.tshape.stem_len * s) / 2)
            self.act_dim = 3 if self.single_spin else 2

    # ------------------------------------------------------------------
    def space(self) -> spaces.Box:
        if self.mode == "kinematic":
            # one [direction, rotation] command for the whole T
            return spaces.Box(
                low=np.array([-math.pi, -1.0], dtype=np.float32),
                high=np.array([math.pi, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        n = self.n_ants
        if self.single_spin:
            low  = np.array([[-math.pi, 0.0, -1.0]], dtype=np.float32)
            high = np.array([[ math.pi, 1.0,  1.0]], dtype=np.float32)
        else:
            low  = np.array([[-math.pi, 0.0]] * n, dtype=np.float32)
            high = np.array([[ math.pi, 1.0]] * n, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    def decode_kinematic(self, actions):
        """Return ``(direction, rotation)`` from a flat ``[direction, rotation]`` action."""
        a = np.asarray(actions, dtype=np.float32).reshape(-1)
        return float(a[0]), float(np.clip(a[1], -1.0, 1.0))

    def to_wrench(self, actions, state):
        """(dynamic mode) Sum agent action(s) into a net (force, torque)."""
        actions = np.asarray(actions, dtype=np.float32)  # (n_ants, act_dim)
        total_force = np.zeros(2, dtype=np.float32)
        total_torque = 0.0
        for i in range(self.n_ants):
            angle = float(actions[i, 0])
            magnitude = float(np.clip(actions[i, 1], 0.0, 1.0))
            if magnitude >= 1e-8:
                move_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                force = self.push_strength * magnitude * move_dir
                arm = state.attachment_world(i) - state.object_center
                total_force += force
                total_torque += arm[0] * force[1] - arm[1] * force[0]
            if self.single_spin:
                spin = float(np.clip(actions[i, 2], -1.0, 1.0))
                total_torque += spin * self.spin_strength
        return total_force, total_torque
