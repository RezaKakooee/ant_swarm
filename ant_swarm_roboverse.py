"""Ant swarm pushing an I-shape — RoboVerse port of new_exps/t_barrier_swarm_env.py.

Strategy: the original env already implements a complete 2D rigid-body
simulation in Python (custom collision/forces/integration). Re-implementing
those contact dynamics with a backend simulator would require putting a
controllable robot at every ant, which doesn't scale to 100+ ants.

Instead, we run the original env unchanged as the source of truth and use
RoboVerse purely as the 3D renderer:

  1. Python env steps  →  computes new ant and I-shape poses.
  2. handler.set_states(new poses)  →  pushes those poses into the scenario.
  3. handler.simulate() + handler.get_states()  →  one render call.
  4. ObsSaver writes the camera frame to an mp4.

Only --sim mujoco is supported (kinematic-playback only — no backend physics
needed, but mujoco is the most permissive backend for many tiny bodies).

Usage:
    python new_exps/ant_swarm_roboverse.py --headless
    python new_exps/ant_swarm_roboverse.py --headless --n-ants 100 --n-steps 400
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler


# ---------------------------------------------------------------------------
# Original 2D physics (verbatim from t_barrier_swarm_env.py, minus render())
# ---------------------------------------------------------------------------

@dataclass
class LocalRect:
    center: np.ndarray
    half_size: np.ndarray


def _tshape_overlaps_walls(
    center: np.ndarray,
    angle: float,
    rects: list,
    walls_aabb: list,
    world_size: tuple,
    margin: float = 0.0,
) -> bool:
    """AABB overlap test: T-shape at (center, angle) vs walls and world boundary."""
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    W, H = world_size
    for rect in rects:
        hx, hy = float(rect.half_size[0]), float(rect.half_size[1])
        lx, ly = float(rect.center[0]), float(rect.center[1])
        corners = np.array([
            [lx - hx, ly - hy], [lx + hx, ly - hy],
            [lx + hx, ly + hy], [lx - hx, ly + hy],
        ])
        wc = (R @ corners.T).T + center
        txmin, txmax = float(wc[:, 0].min()), float(wc[:, 0].max())
        tymin, tymax = float(wc[:, 1].min()), float(wc[:, 1].max())
        # World boundary check
        if txmin < margin or txmax > W - margin or tymin < margin or tymax > H - margin:
            return True
        # Wall overlap check
        for xmin, xmax, ymin, ymax in walls_aabb:
            if txmin < xmax and txmax > xmin and tymin < ymax and tymax > ymin:
                return True
    return False


def sample_tshape_pose(
    rects: list,
    walls_aabb: list,
    world_size: tuple,
    *,
    rng: np.random.Generator | None = None,
    angle_range: tuple = (-math.pi / 2, math.pi / 2),
    margin: float = 0.06,
    max_tries: int = 500,
) -> tuple[np.ndarray, float]:
    """Sample a random (center, angle) that fits inside the world and clears all walls.

    Args:
        rects:       Scaled T-shape collision rects.
        walls_aabb:  Scaled wall AABBs [(xmin, xmax, ymin, ymax), ...].
        world_size:  (W, H) in scaled metres.
        rng:         NumPy random generator; created fresh if None.
        angle_range: (min, max) angle in radians to sample from.
        margin:      Minimum clearance from world boundary in metres.
        max_tries:   Rejection-sample attempts before falling back to a safe default.

    Returns:
        (center, angle) in scaled world coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()
    W, H = world_size
    for _ in range(max_tries):
        angle = float(rng.uniform(*angle_range))
        center = np.array([rng.uniform(margin, W - margin),
                           rng.uniform(margin, H - margin)], dtype=np.float32)
        if not _tshape_overlaps_walls(center, angle, rects, walls_aabb, world_size, margin):
            return center, angle
    log.warning("sample_tshape_pose: no valid pose in %d tries — using safe default.", max_tries)
    return np.array([0.22 * W, 0.36 * H / 0.72], dtype=np.float32), math.radians(-35)


class MovableTShape:
    """Rigid movable T-shaped object + scenario configuration."""

    def __init__(
        self,
        center=None,   # (x, y) pre-scale coords, or None → sampled randomly
        angle=None,    # radians, or None → sampled randomly
        *,
        # T-shape geometry — all thicknesses share one value.
        stem_len=0.30,
        cap_big_len=0.18,    # longer cap (left end of stem)
        cap_small_len=0.09,  # shorter cap (right end of stem)
        thickness=0.02,      # uniform thickness: stem, both caps, and all walls
        tshape_z=0.02,       # z of T-shape body origin (= floor contact point)
        tshape_height=0.04,  # T-shape extrusion height (used in MJCF and for ant_z)
        # World
        world_size=(1.0, 0.72),
        # Walls — list of (x, y) centers of each vertical wall segment.
        # Each segment is wall_len long and `thickness` thick.
        wall_len=0.30,
        wall_height=0.08,
        wall_positions=None,   # default: two gates at x=0.46 and x=0.67
        # Extra length added outward (away from the passage) for 3D rendering only,
        # so walls visually reach the scene boundary.  Does not affect 2D physics.
        wall_render_extra=0.20,
        # Ants — permanently attached to the T-shape, 10 workers each with real ant-like mass.
        n_ants=10,
        ant_radius=0.012,
        # Ants are beside the T-shape (same height as its centre).
        # tshape_z=0.02, tshape_height=0.04 → centre at 0.04.
        ant_z=0.04,
        ant_mass=0.001,       # ~1 mg per ant, like a real ant
        # Target terminal velocity ≈ 0.001 m/step  →  force = vel × mass × (1-friction)
        # = 0.001 × 0.5 × 0.04 = 2e-5 total  →  2e-6 per ant.
        push_strength=0.000002,
        # Physics
        object_mass=0.5,
        object_inertia=0.04,
        linear_friction=0.96,
        angular_friction=0.94,
        # Task
        goal=(0.20, 0.54),
        spawn_center=(0.75, 0.25),
        scene_scale: float = 1.0,
        pose_seed=None,        # RNG seed used when center or angle is None
    ):
        s = scene_scale
        self.scene_scale = s

        # ----------------------------------------------------------------
        # 1. Geometry (must be computed BEFORE pose so sampling can use it)
        # ----------------------------------------------------------------
        self.stem_len = stem_len * s
        self.thickness = thickness * s
        self.cap_big_len = cap_big_len * s
        self.cap_small_len = cap_small_len * s
        self.tshape_z = tshape_z * s
        self.tshape_height = tshape_height * s

        # World (scaled)
        self.world_size = (world_size[0] * s, world_size[1] * s)
        W, H = self.world_size

        # Walls (scaled)
        self.wall_len = wall_len * s
        self.wall_render_extra = wall_render_extra * s
        self.wall_height = wall_height * s
        if wall_positions is not None:
            self.wall_positions = [(wx * s, wy * s) for wx, wy in wall_positions]
        else:
            hl = self.wall_len / 2
            upper_cy = H - hl
            lower_cy = hl
            self.wall_positions = [
                (0.46 * s, upper_cy),  # left gate — upper
                (0.46 * s, lower_cy),  # left gate — lower
                (0.67 * s, upper_cy),  # right gate — upper
                (0.67 * s, lower_cy),  # right gate — lower
            ]

        # Collision rects (in T-shape local frame, scaled)
        ht = self.thickness / 2
        self.rects = [
            LocalRect(np.array([0.0, 0.0], dtype=np.float32),
                      np.array([self.stem_len / 2, ht], dtype=np.float32)),
            LocalRect(np.array([-self.stem_len / 2, 0.0], dtype=np.float32),
                      np.array([ht, self.cap_big_len / 2], dtype=np.float32)),
            LocalRect(np.array([self.stem_len / 2, 0.0], dtype=np.float32),
                      np.array([ht, self.cap_small_len / 2], dtype=np.float32)),
        ]

        # ----------------------------------------------------------------
        # 2. Resolve initial pose — sample if center or angle is None
        # ----------------------------------------------------------------
        _walls_aabb = [
            (wx - ht, wx + ht, wy - self.wall_len / 2, wy + self.wall_len / 2)
            for wx, wy in self.wall_positions
        ]
        if center is None or angle is None:
            _rng = np.random.default_rng(pose_seed)
            _c, _a = sample_tshape_pose(
                self.rects, _walls_aabb, self.world_size, rng=_rng
            )
            _center = _c if center is None else np.array(center, dtype=np.float32) * s
            _angle  = _a if angle  is None else float(angle)
        else:
            _center = np.array(center, dtype=np.float32) * s
            _angle  = float(angle)

        # ----------------------------------------------------------------
        # 3. Dynamic state + stored initial pose (for reset)
        # ----------------------------------------------------------------
        self.center = np.asarray(_center, dtype=np.float32)
        self.angle  = float(_angle)
        self.vel     = np.zeros(2, dtype=np.float32)
        self.ang_vel = 0.0
        self.init_center = self.center.copy()
        self.init_angle  = self.angle

        # ----------------------------------------------------------------
        # 4. Entities and physics
        # ----------------------------------------------------------------
        self.n_ants = n_ants
        self.ant_radius = ant_radius * s
        self.ant_z = ant_z * s
        self.ant_mass = ant_mass

        # Forces scale with s; inertia scales as s^2.
        self.push_strength = push_strength * s
        self.object_mass = object_mass
        self.object_inertia = object_inertia * s * s
        self.linear_friction = linear_friction
        self.angular_friction = angular_friction

        # Spawn / goal (scaled)
        self.goal = np.array(goal, dtype=np.float32) * s
        self.spawn_center = np.array(spawn_center, dtype=np.float32) * s

    def copy_with_pose(self, center, angle):
        """Return a shallow copy with fresh pose (for env.reset)."""
        import copy
        new = copy.copy(self)
        new.center = np.array(center, dtype=np.float32)
        new.angle = float(angle)
        new.vel = np.zeros(2, dtype=np.float32)
        new.ang_vel = 0.0
        return new

    def rot(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, -s], [s, c]], dtype=np.float32)

    def world_to_local(self, p):
        return self.rot().T @ (p - self.center)

    def local_to_world(self, p):
        return self.center + self.rot() @ p

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

    def distance_gradient(self, p):
        eps = 1e-4
        gx = self.distance(p + np.array([eps, 0])) - self.distance(p - np.array([eps, 0]))
        gy = self.distance(p + np.array([0, eps])) - self.distance(p - np.array([0, eps]))
        g = np.array([gx, gy], dtype=np.float32)
        n = np.linalg.norm(g)
        if n < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return g / n

    def local_corners(self):
        corners = []
        for r in self.rects:
            hx, hy = r.half_size
            cx, cy = r.center
            corners += [
                [cx - hx, cy - hy],
                [cx - hx, cy + hy],
                [cx + hx, cy - hy],
                [cx + hx, cy + hy],
            ]
        return np.array(corners, dtype=np.float32)

    def world_corners(self):
        R = self.rot()
        return self.center[None, :] + self.local_corners() @ R.T


class AntMovableTShapeEnv:
    """10 ants permanently attached to the T-shape.

    Each ant occupies a fixed point on the T-shape surface (local frame).
    At every step each ant applies a force at its attachment point; the
    combined forces and torques drive the T-shape dynamics.  Ants ride the
    T-shape — they never detach and never move independently.
    """

    def __init__(self, cfg: MovableTShape | None = None, seed=None):
        self.cfg = cfg if cfg is not None else MovableTShape()
        c = self.cfg

        self.n_ants = c.n_ants
        self.W, self.H = c.world_size
        self.ant_radius = c.ant_radius
        self.push_strength = c.push_strength
        self.object_mass = c.object_mass
        self.object_inertia = c.object_inertia
        self.linear_friction = c.linear_friction
        self.angular_friction = c.angular_friction
        self.rng = np.random.default_rng(seed)

        self.goal = c.goal.copy()
        self.obj: MovableTShape | None = None
        self.ants: np.ndarray | None = None  # world positions, derived from obj pose

        # Attachment offsets in T-shape local frame — fixed for the lifetime of the env.
        self.attachment_offsets = self._make_attachment_offsets()

        # Wall AABBs (xmin, xmax, ymin, ymax) — one entry per wall_position.
        ht = c.thickness / 2
        hl = c.wall_len / 2
        self.walls_aabb: list[tuple[float, float, float, float]] = [
            (wx - ht, wx + ht, wy - hl, wy + hl)
            for wx, wy in c.wall_positions
        ]

    def _make_attachment_offsets(self) -> np.ndarray:
        """Sample n_ants points on the T-shape PERIMETER (local frame).

        Each point is on one of the four edges of one of the three rects,
        so ants sit touching the side of the T-shape rather than on top.
        """
        rects = self.cfg.rects
        # Weight each rect by its perimeter so sampling is uniform over length.
        perims = [4.0 * (float(r.half_size[0]) + float(r.half_size[1])) for r in rects]
        total = sum(perims)
        probs = [p / total for p in perims]
        offsets = []
        for _ in range(self.n_ants):
            r_idx = int(self.rng.choice(len(rects), p=probs))
            rect = rects[r_idx]
            cx, cy = float(rect.center[0]), float(rect.center[1])
            hx, hy = float(rect.half_size[0]), float(rect.half_size[1])
            # Parameterise perimeter: bottom → right → top → left
            perim = 2 * (2*hx + 2*hy)  # = 4*(hx+hy)
            t = self.rng.uniform(0, perim)
            if t < 2*hx:
                x, y = cx - hx + t,   cy - hy
            elif t < 2*hx + 2*hy:
                x, y = cx + hx,       cy - hy + (t - 2*hx)
            elif t < 4*hx + 2*hy:
                x, y = cx + hx - (t - 2*hx - 2*hy), cy + hy
            else:
                x, y = cx - hx,       cy + hy - (t - 4*hx - 2*hy)
            offsets.append([x, y])
        return np.array(offsets, dtype=np.float32)

    def _update_ant_positions(self) -> None:
        """Recompute ant world positions from current T-shape pose."""
        self.ants = np.array(
            [self.obj.local_to_world(self.attachment_offsets[i]) for i in range(self.n_ants)],
            dtype=np.float32,
        )

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.obj = self.cfg.copy_with_pose(self.cfg.init_center, self.cfg.init_angle)
        self._update_ant_positions()

    def _t_overlaps_walls(self) -> bool:
        """True if any of the T-shape's three rectangles overlaps a wall."""
        corners = self.obj.world_corners()  # (12, 2) — 4 corners per rect
        for i in range(0, 12, 4):
            c = corners[i:i + 4]
            txmin, txmax = float(c[:, 0].min()), float(c[:, 0].max())
            tymin, tymax = float(c[:, 1].min()), float(c[:, 1].max())
            for xmin, xmax, ymin, ymax in self.walls_aabb:
                if txmin < xmax and txmax > xmin and tymin < ymax and tymax > ymin:
                    return True
        return False

    def step(self, actions):
        """Apply per-ant forces at attachment points, then integrate T-shape dynamics."""
        actions = np.asarray(actions, dtype=np.int64)
        dirs = np.array([
            [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0],
            [-1, -1], [1, -1], [-1, 1], [1, 1],
        ], dtype=np.float32)
        dirs[5:] /= np.sqrt(2)

        total_force = np.zeros(2, dtype=np.float32)
        total_torque = 0.0

        for i in range(self.n_ants):
            move_dir = dirs[actions[i]]
            if np.linalg.norm(move_dir) < 1e-8:
                continue
            force = self.push_strength * move_dir
            attachment_world = self.obj.local_to_world(self.attachment_offsets[i])
            arm = attachment_world - self.obj.center
            total_force += force
            total_torque += arm[0] * force[1] - arm[1] * force[0]

        self._integrate_object(total_force, total_torque)
        self._update_ant_positions()

    def _integrate_object(self, force, torque):
        self.obj.vel = self.linear_friction * self.obj.vel + force / self.object_mass
        self.obj.ang_vel = self.angular_friction * self.obj.ang_vel + torque / self.object_inertia

        # ------------------------------------------------------------------
        # Sub-step integration so the T-shape cannot tunnel through thin
        # walls.  We split the full displacement into ~10 micro-steps and
        # back out as soon as an overlap is detected.
        # ------------------------------------------------------------------
        n_substeps = 10
        dp = self.obj.vel / n_substeps
        da = self.obj.ang_vel / n_substeps

        for _ in range(n_substeps):
            prev_center = self.obj.center.copy()
            prev_angle = self.obj.angle
            self.obj.center = prev_center + dp
            self.obj.angle = prev_angle + da

            if self._t_overlaps_walls():
                # Try sliding along x
                self.obj.center = np.array([prev_center[0] + dp[0], prev_center[1]],
                                           dtype=np.float32)
                if self._t_overlaps_walls():
                    # Try sliding along y
                    self.obj.center = np.array([prev_center[0], prev_center[1] + dp[1]],
                                               dtype=np.float32)
                    if self._t_overlaps_walls():
                        # Fully blocked — revert and damp.
                        self.obj.center = prev_center
                        self.obj.angle = prev_angle
                        self.obj.vel *= -0.2
                        self.obj.ang_vel *= -0.2
                        break
                    else:
                        self.obj.vel[0] *= -0.2
                else:
                    self.obj.vel[1] *= -0.2

            if self._t_overlaps_walls():
                self.obj.angle = prev_angle
                self.obj.ang_vel *= -0.2

        # Soft world boundaries
        corners = self.obj.world_corners()
        shift = np.zeros(2, dtype=np.float32)
        margin = 0.025 * self.cfg.scene_scale
        if corners[:, 0].min() < margin:
            shift[0] += margin - corners[:, 0].min()
            self.obj.vel[0] *= -0.15
        if corners[:, 0].max() > self.W - margin:
            shift[0] -= corners[:, 0].max() - (self.W - margin)
            self.obj.vel[0] *= -0.15
        if corners[:, 1].min() < margin:
            shift[1] += margin - corners[:, 1].min()
            self.obj.vel[1] *= -0.15
        if corners[:, 1].max() > self.H - margin:
            shift[1] -= corners[:, 1].max() - (self.H - margin)
            self.obj.vel[1] *= -0.15
        self.obj.center += shift

    def heuristic_actions(self):
        """Each ant pushes in its own direction — biased toward the goal but with
        large individual noise, so ants genuinely disagree: one pushes right,
        another up, another toward the goal, etc.  The net average still moves
        the T-shape toward the goal while creating realistic rotational dynamics.
        """
        obj_to_goal = self.goal - self.obj.center
        desired = obj_to_goal / max(np.linalg.norm(obj_to_goal), 1e-8)
        actions = np.zeros(self.n_ants, dtype=np.int64)
        for i in range(self.n_ants):
            # sigma=0.7 gives genuine direction diversity: some ants push 90°+
            # off from the goal direction.
            v = desired + self.rng.normal(0, 0.7, size=2)
            v = v.astype(np.float32)
            actions[i] = self._vector_to_action(v / max(float(np.linalg.norm(v)), 1e-8))
        return actions

    def _vector_to_action(self, v):
        dirs = np.array([
            [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0],
            [-1, -1], [1, -1], [-1, 1], [1, 1],
        ], dtype=np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = np.divide(dirs, np.maximum(norms, 1e-6))
        return int(np.argmax(dirs @ v))


# ---------------------------------------------------------------------------
# I-shape MJCF generator
# ---------------------------------------------------------------------------

def build_tshape_mjcf(
    stem_len: float,
    cap_big_len: float,
    cap_small_len: float,
    thickness: float,
    height: float = 0.04,
) -> str:
    """Three boxes welded into one rigid body lying flat on the floor.

    Root is at the T-shape's geometric center. All dims use the same
    ``thickness`` (stem width, cap depth, wall thickness are all equal).
    """
    ht = thickness / 2
    return f"""<mujoco model="t_shape">
    <worldbody>
        <body name="root" pos="0 0 0">
            <geom name="stem" type="box"
                  pos="0 0 {height / 2}"
                  size="{stem_len / 2} {ht} {height / 2}"
                  rgba="0.67 0.08 0.055 1" mass="0.3"/>
            <geom name="cap_big" type="box"
                  pos="{-stem_len / 2} 0 {height / 2}"
                  size="{ht} {cap_big_len / 2} {height / 2}"
                  rgba="0.67 0.08 0.055 1" mass="0.1"/>
            <geom name="cap_small" type="box"
                  pos="{stem_len / 2} 0 {height / 2}"
                  size="{ht} {cap_small_len / 2} {height / 2}"
                  rgba="0.67 0.08 0.055 1" mass="0.1"/>
        </body>
    </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Helpers to convert env state → set_states dict
# ---------------------------------------------------------------------------

def angle_to_quat_wxyz(angle: float) -> list[float]:
    """Rotation around the world z-axis as a wxyz unit quaternion."""
    return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def env_to_states(env: AntMovableTShapeEnv) -> dict:
    """Build the dict that handler.set_states expects from env's 2D state.

    The mujoco handler unconditionally indexes both ``state["objects"]`` and
    ``state["robots"]``, so the empty robots dict has to be present even
    when the scenario contains no robots.

    Walls are static, but we still push their pose every step so the very
    first set_states() call (which happens before the loop) covers them.
    """
    cfg = env.cfg
    wall_z = cfg.wall_height / 2
    half_H = cfg.world_size[1] / 2
    extra = cfg.wall_render_extra
    wall_states = {}
    for j, (wx, wy) in enumerate(cfg.wall_positions):
        # Shift center outward by extra/2 so the outer edge extends past the
        # scene boundary while the inner (passage-side) edge stays fixed.
        render_wy = wy + extra / 2 if wy > half_H else wy - extra / 2
        wall_states[f"wall_{j}"] = {
            "pos": torch.tensor([wx, render_wy, wall_z], dtype=torch.float32),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        }

    return {
        "robots": {},
        "objects": {
            "i_shape": {
                "pos": torch.tensor(
                    [float(env.obj.center[0]), float(env.obj.center[1]), cfg.tshape_z],
                    dtype=torch.float32,
                ),
                "rot": torch.tensor(
                    angle_to_quat_wxyz(env.obj.angle), dtype=torch.float32
                ),
            },
            **wall_states,
            **{
                f"ant_{i}": {
                    "pos": torch.tensor(
                        [float(env.ants[i, 0]), float(env.ants[i, 1]), cfg.ant_z],
                        dtype=torch.float32,
                    ),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                }
                for i in range(env.n_ants)
            },
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the ant-swarm demo."""

        # Custom MJCF for the I-shape means USD/URDF aren't available,
        # so only mujoco is supported.
        sim: Literal["mujoco"] = "mujoco"
        n_ants: int = 10
        n_steps: int = 260
        seed: int = 8
        # Capture every N env steps. n_steps / frame_every = video frames.
        frame_every: int = 1
        num_envs: int = 1
        headless: bool = True
        # Uniform scale for all scene dimensions (world, shapes, ants, camera).
        # Increase to make the scene physically larger; 1.0 = default 1 m x 1 m world.
        scene_scale: float = 1.0

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

    # ------------------------------------------------------------------
    # Scenario config lives in MovableTShape (env geometry + physics).
    # ------------------------------------------------------------------
    cfg = MovableTShape(n_ants=args.n_ants, scene_scale=args.scene_scale)

    # ------------------------------------------------------------------
    # Run the python env to drive the 2D physics. We do NOT use backend
    # physics here — RoboVerse is only the renderer.
    # ------------------------------------------------------------------
    env = AntMovableTShapeEnv(cfg=cfg, seed=args.seed)
    env.reset(seed=args.seed)

    # ------------------------------------------------------------------
    # Generate the I-shape MJCF (matches env.obj's dimensions exactly).
    # ------------------------------------------------------------------
    mjcf_xml = build_tshape_mjcf(
        stem_len=env.obj.stem_len,
        cap_big_len=env.obj.cap_big_len,
        cap_small_len=env.obj.cap_small_len,
        thickness=env.obj.thickness,
        height=env.obj.tshape_height,
    )
    mjcf_dir = os.path.abspath("new_exps/output/ant_swarm")
    os.makedirs(mjcf_dir, exist_ok=True)
    mjcf_path = os.path.join(mjcf_dir, "i_shape.xml")
    with open(mjcf_path, "w") as f:
        f.write(mjcf_xml)
    log.info(f"Wrote I-shape MJCF → {mjcf_path}")

    # ------------------------------------------------------------------
    # Scenario. No robots — every entity is just an Object.
    # ------------------------------------------------------------------
    scenario = ScenarioCfg(
        robots=[],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )
    # Top-down camera scaled with the world.  scene_scale=1 → camera at 1.6 m;
    # larger worlds push the camera up proportionally to keep the full arena in frame.
    cx, cy = env.W / 2, env.H / 2
    s = args.scene_scale
    scenario.cameras = [
        PinholeCameraCfg(
            name="topdown",
            width=1280,
            height=820,
            pos=(cx, cy - 0.05 * s, 1.6 * s),
            look_at=(cx, cy, 0.0),
        )
    ]

    # Objects: T-shape (rigid, custom MJCF), wall cubes (one per wall_position),
    # and one sphere per ant.
    # 3D wall size: extend outward by wall_render_extra so walls visually touch the
    # scene boundary.  Physics collision uses cfg.wall_positions (unmodified).
    render_wall_len = cfg.wall_len + cfg.wall_render_extra
    wall_objects: list[PrimitiveCubeCfg] = [
        PrimitiveCubeCfg(
            name=f"wall_{j}",
            size=(cfg.thickness, render_wall_len, cfg.wall_height),
            color=[0.55, 0.55, 0.58],
            physics=PhysicStateType.RIGIDBODY,
        )
        for j in range(len(cfg.wall_positions))
    ]

    scenario.objects = [
        RigidObjCfg(
            name="i_shape",
            mjcf_path=mjcf_path,
            usd_path=None,
            urdf_path=None,
            physics=PhysicStateType.RIGIDBODY,
        ),
    ] + wall_objects + [
        PrimitiveSphereCfg(
            name=f"ant_{i}",
            radius=cfg.ant_radius,
            color=[0.10, 0.25, 0.75],
            physics=PhysicStateType.RIGIDBODY,
        )
        for i in range(cfg.n_ants)
    ]

    log.info(f"Using simulator: {args.sim}")
    handler = get_handler(scenario)

    # Initial state — everything in env-space coordinates.
    init = env_to_states(env)
    handler.set_states([init] * scenario.num_envs)
    handler.simulate()
    obs = handler.get_states(mode="tensor")

    os.makedirs("new_exps/output", exist_ok=True)
    obs_saver = ObsSaver(
        video_path=f"new_exps/output/ant_swarm_{args.sim}.mp4"
    )
    obs_saver.add(obs)

    # ------------------------------------------------------------------
    # Main loop: python env step → push state to handler → capture frame.
    # ------------------------------------------------------------------
    for step in range(args.n_steps):
        env.step(env.heuristic_actions())
        handler.set_states([env_to_states(env)] * scenario.num_envs)
        handler.simulate()
        obs = handler.get_states(mode="tensor")
        if (step + 1) % args.frame_every == 0:
            obs_saver.add(obs)
        if (step + 1) % 40 == 0:
            d = float(np.linalg.norm(env.obj.center - env.goal))
            log.info(
                f"step {step + 1:>4}/{args.n_steps}  "
                f"obj=({env.obj.center[0]:+.2f},{env.obj.center[1]:+.2f})  "
                f"d_to_goal={d:.3f}"
            )

    obs_saver.save()
    handler.close()
