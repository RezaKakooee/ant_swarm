"""Observation component: build the per-ant observation and its space.

Per ant (9 floats, ~[-1, 1]):
    [att_local_x, att_local_y,   # attachment offset in T local frame
     obj_x, obj_y,               # T centre, normalised by world size
     goal_dx, goal_dy,           # (goal - obj), normalised
     sin_angle, cos_angle,       # T orientation
     ang_vel]                    # angular velocity (clipped)
"""
from __future__ import annotations

import math

import numpy as np

from ._gym import spaces


class ObservationModel:
    def __init__(self, cfg, layout, tshape):
        self.n_ants = int(cfg.ants.n)
        self.world_size = layout.world_size
        self.goal = layout.goal
        self.stem_half = tshape.stem_len / 2
        self.cap_half = max(tshape.cap_big_len, tshape.cap_small_len) / 2

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.n_ants, 9), dtype=np.float32)

    def observe(self, state) -> np.ndarray:
        W, H = self.world_size
        obj = state.obj
        offsets = state.attachment_offsets
        obs = np.zeros((self.n_ants, 9), dtype=np.float32)
        goal_d = (self.goal - obj.center) / np.array([W, H], dtype=np.float32)
        for i in range(self.n_ants):
            att = offsets[i]
            obs[i] = [
                float(att[0]) / self.stem_half,
                float(att[1]) / self.cap_half,
                float(obj.center[0]) / W,
                float(obj.center[1]) / H,
                float(goal_d[0]),
                float(goal_d[1]),
                math.sin(obj.angle),
                math.cos(obj.angle),
                float(np.clip(obj.ang_vel / 0.05, -1, 1)),
            ]
        return obs
