"""Observation component: build the per-ant observation and its space.

Per ant (9 base + 2 optional velocity + 16 barrier features, ~[-1, 1]):
    base (9, always present):
      [att_local_x, att_local_y,   # attachment offset in T local frame
       obj_x, obj_y,               # T centre, normalised by world size
       goal_dx, goal_dy,           # (goal - obj), normalised
       sin_angle, cos_angle,       # T orientation
       ang_vel]                    # angular velocity (clipped)
    dynamic velocity (2, dynamic mode only by default):
      [vel_x, vel_y]               # world-frame linear velocity (clipped)
    barrier (16):
      distances from the T-shape's 4 arm tips to the 4 inner wall "heads"
      (the gap-facing wall corners), normalised by the world diagonal.
      Row-major: tip0→head0..3, tip1→head0..3, ...

The barrier block gives the policy a direct "how close is each arm to clipping
a wall corner" signal — the cue needed to learn to rotate near the gap.

Dynamic observations are 27-dimensional.  Set
``env.observe_linear_velocity: false`` to retain the legacy 25-dimensional
layout (for example, when loading an old checkpoint).  Kinematic mode also
uses the 25-dimensional layout because it deliberately has no momentum.
"""
from __future__ import annotations

import math

import numpy as np

from ._gym import spaces

N_BASE = 9
N_LINEAR_VELOCITY = 2
N_TIPS = 4
N_HEADS = 4
N_BARRIER = N_TIPS * N_HEADS
OBS_DIM = N_BASE + N_BARRIER  # legacy / kinematic: 25
DYNAMIC_OBS_DIM = OBS_DIM + N_LINEAR_VELOCITY  # 27


class ObservationModel:
    def __init__(self, cfg, layout, tshape):
        self.n_ants = int(cfg.ants.n)
        self.world_size = layout.world_size
        self.goal = layout.goal
        self.stem_half = tshape.stem_len / 2
        self.cap_half = max(tshape.cap_big_len, tshape.cap_small_len) / 2

        self.wall_heads = layout.wall_heads          # (4, 2) static
        self.world_diag = layout.world_diag
        # goal vector is measured from the same point the reward tracks
        self.track_local = tshape.track_local_point(getattr(cfg.env, "goal_track", "center"))

        motion = getattr(cfg, "motion", None)
        self.motion_mode = getattr(motion, "mode", "dynamic") if motion else "dynamic"
        requested_velocity = bool(getattr(
            cfg.env, "observe_linear_velocity", self.motion_mode == "dynamic"))
        self.observe_linear_velocity = (
            self.motion_mode == "dynamic" and requested_velocity)
        self.obs_dim = DYNAMIC_OBS_DIM if self.observe_linear_velocity else OBS_DIM

        # Normalise by the theoretical steady-state speed under aligned,
        # full-strength pushes: (total force / mass) / (1 - friction).
        # This follows the actual discrete dynamics and adapts to ant count.
        if self.observe_linear_velocity:
            ph = cfg.physics
            scene_scale = float(cfg.scene_scale)
            total_force = abs(float(ph.push_strength)) * scene_scale * self.n_ants
            mass = max(abs(float(ph.object_mass)), 1e-8)
            damping = max(1.0 - min(abs(float(ph.linear_friction)), 0.999), 1e-3)
            self.linear_velocity_scale = max(total_force / mass / damping, 1e-8)

        # The T's four arm tips in local frame (big-cap ±, small-cap ±).
        sb, cb, cs = tshape.stem_len / 2, tshape.cap_big_len / 2, tshape.cap_small_len / 2
        self.tip_local = np.array([
            [-sb,  cb], [-sb, -cb],   # big-cap top / bottom
            [ sb,  cs], [ sb, -cs],   # small-cap top / bottom
        ], dtype=np.float32)

    def space(self) -> spaces.Box:
        return spaces.Box(
            -1.0, 1.0, shape=(self.n_ants, self.obs_dim), dtype=np.float32)

    def _barrier_features(self, obj) -> np.ndarray:
        """16 normalised tip→head distances for the current T pose."""
        tips_world = obj.center[None, :] + self.tip_local @ obj.rot().T   # (4, 2)
        # pairwise distances tip (4) × head (4) → (4, 4)
        diff = tips_world[:, None, :] - self.wall_heads[None, :, :]
        dists = np.linalg.norm(diff, axis=2) / self.world_diag
        return dists.reshape(-1).astype(np.float32)                       # (16,)

    def observe(self, state) -> np.ndarray:
        W, H = self.world_size
        obj = state.obj
        offsets = state.attachment_offsets
        tracked = obj.local_to_world(self.track_local)         # e.g. big-cap centre
        goal_d = (self.goal - tracked) / np.array([W, H], dtype=np.float32)
        barrier = self._barrier_features(obj)   # shared across ants (depends on T pose)

        obs = np.zeros((self.n_ants, self.obs_dim), dtype=np.float32)
        for i in range(self.n_ants):
            att = offsets[i]
            obs[i, :N_BASE] = [
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
            barrier_start = N_BASE
            if self.observe_linear_velocity:
                obs[i, N_BASE:N_BASE + N_LINEAR_VELOCITY] = np.clip(
                    obj.vel / self.linear_velocity_scale, -1.0, 1.0)
                barrier_start += N_LINEAR_VELOCITY
            obs[i, barrier_start:] = barrier
        return obs
