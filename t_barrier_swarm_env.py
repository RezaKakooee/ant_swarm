"""Ant swarm pushing a T-shaped object through barrier gates — pure-Python env.

Physics and rendering are entirely in Python (NumPy + Matplotlib).
No MuJoCo / RoboVerse dependency.  This file is the fast-iteration sibling of
``ant_swarm_roboverse.py`` which uses MuJoCo for 3-D rendering.

The environment is Gymnasium-compatible (``gym.Env``) so it can be used for RL
as-is.  The ``make_demo_gif`` function at the bottom generates an animated GIF
without any additional dependencies.

Usage:
    python new_exps/t_barrier_swarm_env.py          # saves GIF → output/
    python new_exps/t_barrier_swarm_env.py --steps 400 --seed 3
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

@dataclass
class LocalRect:
    center: np.ndarray
    half_size: np.ndarray


# ---------------------------------------------------------------------------
# Pose sampling helpers
# ---------------------------------------------------------------------------

def _tshape_overlaps_walls(
    center: np.ndarray,
    angle: float,
    rects: list,
    walls_aabb: list,
    world_size: tuple,
    margin: float = 0.0,
) -> bool:
    """AABB overlap: T-shape at (center, angle) vs walls and world boundary."""
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
        if txmin < margin or txmax > W - margin or tymin < margin or tymax > H - margin:
            return True
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
    """Sample a random (center, angle) that clears all walls and world boundary.

    Returns center in *scaled* world coordinates.
    Falls back to a hard-coded safe default if no valid pose found.
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


# ---------------------------------------------------------------------------
# T-shape rigid body
# ---------------------------------------------------------------------------

class MovableTShape:
    """Rigid T-shaped object: geometry, collision, and pose."""

    def __init__(
        self,
        center=None,       # (x, y) pre-scale, or None → sampled randomly
        angle=None,        # radians, or None → sampled randomly
        *,
        stem_len=0.30,
        cap_big_len=0.18,
        cap_small_len=0.09,
        thickness=0.02,
        tshape_z=0.02,
        tshape_height=0.04,
        world_size=(1.0, 0.72),
        wall_len=0.30,
        wall_height=0.08,
        wall_positions=None,
        wall_render_extra=0.20,
        n_ants=10,
        ant_radius=0.012,
        ant_z=0.04,
        ant_mass=0.001,
        push_strength=0.000002,
        object_mass=0.5,
        object_inertia=0.04,
        linear_friction=0.96,
        angular_friction=0.94,
        goal=(0.20, 0.54),
        spawn_center=(0.75, 0.25),
        scene_scale: float = 1.0,
        pose_seed=None,
    ):
        s = scene_scale
        self.scene_scale = s

        # --- geometry (scaled) ---
        self.stem_len = stem_len * s
        self.thickness = thickness * s
        self.cap_big_len = cap_big_len * s
        self.cap_small_len = cap_small_len * s
        self.tshape_z = tshape_z * s
        self.tshape_height = tshape_height * s

        self.world_size = (world_size[0] * s, world_size[1] * s)
        W, H = self.world_size

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
                (0.46 * s, upper_cy),
                (0.46 * s, lower_cy),
                (0.67 * s, upper_cy),
                (0.67 * s, lower_cy),
            ]

        ht = self.thickness / 2
        self.rects = [
            LocalRect(np.array([0.0, 0.0], dtype=np.float32),
                      np.array([self.stem_len / 2, ht], dtype=np.float32)),
            LocalRect(np.array([-self.stem_len / 2, 0.0], dtype=np.float32),
                      np.array([ht, self.cap_big_len / 2], dtype=np.float32)),
            LocalRect(np.array([self.stem_len / 2, 0.0], dtype=np.float32),
                      np.array([ht, self.cap_small_len / 2], dtype=np.float32)),
        ]

        _walls_aabb = [
            (wx - ht, wx + ht, wy - self.wall_len / 2, wy + self.wall_len / 2)
            for wx, wy in self.wall_positions
        ]

        # --- resolve initial pose ---
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

        self.center = np.asarray(_center, dtype=np.float32)
        self.angle  = float(_angle)
        self.vel     = np.zeros(2, dtype=np.float32)
        self.ang_vel = 0.0
        self.init_center = self.center.copy()
        self.init_angle  = self.angle

        # --- entities & physics ---
        self.n_ants = n_ants
        self.ant_radius = ant_radius * s
        self.ant_z = ant_z * s
        self.ant_mass = ant_mass
        self.push_strength = push_strength * s
        self.object_mass = object_mass
        self.object_inertia = object_inertia * s * s
        self.linear_friction = linear_friction
        self.angular_friction = angular_friction
        self.goal = np.array(goal, dtype=np.float32) * s
        self.spawn_center = np.array(spawn_center, dtype=np.float32) * s

    def copy_with_pose(self, center, angle):
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
        return np.zeros(2, dtype=np.float32) if n < 1e-8 else g / n

    def local_corners(self):
        corners = []
        for r in self.rects:
            hx, hy = r.half_size
            cx, cy = r.center
            corners += [[cx-hx, cy-hy], [cx-hx, cy+hy], [cx+hx, cy-hy], [cx+hx, cy+hy]]
        return np.array(corners, dtype=np.float32)

    def world_corners(self):
        return self.center[None, :] + self.local_corners() @ self.rot().T


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AntSwarmEnv(gym.Env):
    """Gymnasium env: 10 ants attached to a T-shape, pushing it toward a goal.

    Observation per ant (9 floats, normalised to roughly [-1, 1]):
        att_local_x, att_local_y  — attachment offset in T-shape local frame
        obj_x, obj_y              — T-shape centre, normalised by world size
        goal_dx, goal_dy          — (goal − obj_centre), normalised
        sin_angle, cos_angle      — T-shape orientation
        ang_vel                   — angular velocity (clipped)

    Action per ant: discrete 9 (stay / 4 cardinal / 4 diagonal).
    Reward: shared progress-based (obj centre moves toward goal).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg: MovableTShape | None = None,
        max_steps: int = 400,
        render_mode: str = "rgb_array",
        seed=None,
    ):
        super().__init__()
        self.cfg = cfg if cfg is not None else MovableTShape()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        n = self.cfg.n_ants
        self.action_space = spaces.MultiDiscrete([9] * n)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n, 9), dtype=np.float32)

        self.obj: MovableTShape | None = None
        self.ants: np.ndarray | None = None
        self.attachment_offsets = self._make_attachment_offsets()
        self.step_count = 0
        self._prev_dist = 0.0

        c = self.cfg
        ht = c.thickness / 2
        hl = c.wall_len / 2
        self.walls_aabb = [
            (wx - ht, wx + ht, wy - hl, wy + hl)
            for wx, wy in c.wall_positions
        ]

    # ------------------------------------------------------------------
    # Attachment offsets — sampled once on the T-shape PERIMETER
    # ------------------------------------------------------------------

    def _make_attachment_offsets(self) -> np.ndarray:
        rects = self.cfg.rects
        perims = [4.0 * (float(r.half_size[0]) + float(r.half_size[1])) for r in rects]
        total = sum(perims)
        probs = [p / total for p in perims]
        offsets = []
        for _ in range(self.cfg.n_ants):
            r_idx = int(self.rng.choice(len(rects), p=probs))
            rect = rects[r_idx]
            cx, cy = float(rect.center[0]), float(rect.center[1])
            hx, hy = float(rect.half_size[0]), float(rect.half_size[1])
            perim = 2 * (2*hx + 2*hy)
            t = self.rng.uniform(0, perim)
            if t < 2*hx:
                x, y = cx - hx + t, cy - hy
            elif t < 2*hx + 2*hy:
                x, y = cx + hx, cy - hy + (t - 2*hx)
            elif t < 4*hx + 2*hy:
                x, y = cx + hx - (t - 2*hx - 2*hy), cy + hy
            else:
                x, y = cx - hx, cy + hy - (t - 4*hx - 2*hy)
            offsets.append([x, y])
        return np.array(offsets, dtype=np.float32)

    def _update_ant_positions(self):
        self.ants = np.array(
            [self.obj.local_to_world(self.attachment_offsets[i]) for i in range(self.cfg.n_ants)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.obj = self.cfg.copy_with_pose(self.cfg.init_center, self.cfg.init_angle)
        self._prev_dist = float(np.linalg.norm(self.obj.center - self.cfg.goal))
        self._update_ant_positions()
        # Each ant gets its own persistent push direction, initialised with large
        # noise so they genuinely disagree from the start.
        obj_to_goal = self.cfg.goal - self.obj.center
        desired = obj_to_goal / max(float(np.linalg.norm(obj_to_goal)), 1e-8)
        noise = self.rng.normal(0, 0.7, size=(self.cfg.n_ants, 2)).astype(np.float32)
        dirs = desired + noise
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        self._ant_dirs = dirs / np.maximum(norms, 1e-8)
        return self._obs(), {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        self.step_count += 1

        dirs = np.array([
            [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0],
            [-1, -1], [1, -1], [-1, 1], [1, 1],
        ], dtype=np.float32)
        dirs[5:] /= np.sqrt(2)

        total_force = np.zeros(2, dtype=np.float32)
        total_torque = 0.0
        for i in range(self.cfg.n_ants):
            move_dir = dirs[actions[i]]
            if np.linalg.norm(move_dir) < 1e-8:
                continue
            force = self.cfg.push_strength * move_dir
            attachment_world = self.obj.local_to_world(self.attachment_offsets[i])
            arm = attachment_world - self.obj.center
            total_force += force
            total_torque += arm[0] * force[1] - arm[1] * force[0]

        self._integrate_object(total_force, total_torque)
        self._update_ant_positions()

        dist = float(np.linalg.norm(self.obj.center - self.cfg.goal))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        reached = dist < 0.05
        reward = np.full(self.cfg.n_ants, 2.0 * progress - 0.001, dtype=np.float32)
        if reached:
            reward += 5.0

        terminated = bool(reached)
        truncated = self.step_count >= self.max_steps
        info = {
            "object_center": self.obj.center.copy(),
            "object_angle": self.obj.angle,
            "object_distance": dist,
            "step": self.step_count,
        }
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _t_overlaps_walls(self) -> bool:
        corners = self.obj.world_corners()
        for i in range(0, 12, 4):
            c = corners[i:i + 4]
            txmin, txmax = float(c[:, 0].min()), float(c[:, 0].max())
            tymin, tymax = float(c[:, 1].min()), float(c[:, 1].max())
            for xmin, xmax, ymin, ymax in self.walls_aabb:
                if txmin < xmax and txmax > xmin and tymin < ymax and tymax > ymin:
                    return True
        return False

    def _integrate_object(self, force, torque):
        c = self.cfg
        self.obj.vel = c.linear_friction * self.obj.vel + force / c.object_mass
        self.obj.ang_vel = c.angular_friction * self.obj.ang_vel + torque / c.object_inertia

        n_sub = 10
        dp = self.obj.vel / n_sub
        da = self.obj.ang_vel / n_sub
        for _ in range(n_sub):
            prev_c = self.obj.center.copy()
            prev_a = self.obj.angle
            self.obj.center = prev_c + dp
            self.obj.angle = prev_a + da
            if self._t_overlaps_walls():
                self.obj.center = np.array([prev_c[0] + dp[0], prev_c[1]], dtype=np.float32)
                if self._t_overlaps_walls():
                    self.obj.center = np.array([prev_c[0], prev_c[1] + dp[1]], dtype=np.float32)
                    if self._t_overlaps_walls():
                        self.obj.center = prev_c
                        self.obj.angle = prev_a
                        self.obj.vel *= -0.2
                        self.obj.ang_vel *= -0.2
                        break
                    else:
                        self.obj.vel[0] *= -0.2
                else:
                    self.obj.vel[1] *= -0.2
            if self._t_overlaps_walls():
                self.obj.angle = prev_a
                self.obj.ang_vel *= -0.2

        W, H = c.world_size
        corners = self.obj.world_corners()
        shift = np.zeros(2, dtype=np.float32)
        m = 0.025 * c.scene_scale
        if corners[:, 0].min() < m:
            shift[0] += m - corners[:, 0].min(); self.obj.vel[0] *= -0.15
        if corners[:, 0].max() > W - m:
            shift[0] -= corners[:, 0].max() - (W - m); self.obj.vel[0] *= -0.15
        if corners[:, 1].min() < m:
            shift[1] += m - corners[:, 1].min(); self.obj.vel[1] *= -0.15
        if corners[:, 1].max() > H - m:
            shift[1] -= corners[:, 1].max() - (H - m); self.obj.vel[1] *= -0.15
        self.obj.center += shift

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _obs(self):
        c = self.cfg
        W, H = c.world_size
        stem_half = c.stem_len / 2
        cap_half = max(c.cap_big_len, c.cap_small_len) / 2
        obs = np.zeros((c.n_ants, 9), dtype=np.float32)
        goal_d = (c.goal - self.obj.center) / np.array([W, H], dtype=np.float32)
        for i in range(c.n_ants):
            att = self.attachment_offsets[i]
            obs[i] = [
                float(att[0]) / stem_half,
                float(att[1]) / cap_half,
                float(self.obj.center[0]) / W,
                float(self.obj.center[1]) / H,
                float(goal_d[0]),
                float(goal_d[1]),
                math.sin(self.obj.angle),
                math.cos(self.obj.angle),
                float(np.clip(self.obj.ang_vel / 0.05, -1, 1)),
            ]
        return obs

    # ------------------------------------------------------------------
    # Heuristic policy
    # ------------------------------------------------------------------

    def heuristic_actions(self) -> np.ndarray:
        """Persistent per-ant directions that drift slowly toward the goal.

        Each ant has its own direction vector (_ant_dirs) set at reset.
        Each step it drifts a little toward the goal and gets a small random
        nudge — so individual ants genuinely push in different directions for
        many consecutive steps rather than re-randomising every frame.
        """
        obj_to_goal = self.cfg.goal - self.obj.center
        desired = obj_to_goal / max(float(np.linalg.norm(obj_to_goal)), 1e-8)
        # Small per-step drift: 5 % pull toward goal + tiny noise.
        drift = 0.05 * desired + self.rng.normal(0, 0.08, size=(self.cfg.n_ants, 2))
        self._ant_dirs = (self._ant_dirs + drift).astype(np.float32)
        norms = np.linalg.norm(self._ant_dirs, axis=1, keepdims=True)
        self._ant_dirs /= np.maximum(norms, 1e-8)
        actions = np.zeros(self.cfg.n_ants, dtype=np.int64)
        for i in range(self.cfg.n_ants):
            actions[i] = self._vector_to_action(self._ant_dirs[i])
        return actions

    def _vector_to_action(self, v: np.ndarray) -> int:
        dirs = np.array([
            [0, 0], [0, -1], [0, 1], [-1, 0], [1, 0],
            [-1, -1], [1, -1], [-1, 1], [1, 1],
        ], dtype=np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = np.divide(dirs, np.maximum(norms, 1e-6))
        return int(np.argmax(dirs @ v))

    # ------------------------------------------------------------------
    # Rendering (pure NumPy, no MuJoCo)
    # ------------------------------------------------------------------

    def render(self):
        c = self.cfg
        W, H = c.world_size
        img_w, img_h = 650, int(650 * H / W)
        img = np.full((img_h, img_w, 3), 0.72, dtype=np.float32)

        def to_px(xy):
            x, y = xy
            return int(x / W * (img_w - 1)), int((H - y) / H * (img_h - 1))

        # --- walls: extend ONLY outward (away from the passage) so the gap stays open ---
        half_H = H / 2
        for wx, wy in c.wall_positions:
            ht_w = c.thickness / 2
            hl = c.wall_len / 2
            extra = c.wall_render_extra
            # Upper segment: extend upward; lower segment: extend downward.
            if wy > half_H:
                y_inner, y_outer = wy - hl, wy + hl + extra
            else:
                y_inner, y_outer = wy + hl, wy - hl - extra
            y_lo, y_hi = min(y_inner, y_outer), max(y_inner, y_outer)
            x0, y0 = to_px((wx - ht_w, y_lo))
            x1, y1 = to_px((wx + ht_w, y_hi))
            xs0 = max(0, min(x0, x1))
            xs1 = min(img_w, max(x0, x1) + 1)
            ys0 = max(0, min(y0, y1))
            ys1 = min(img_h, max(y0, y1) + 1)
            img[ys0:ys1, xs0:xs1] = [0.38, 0.38, 0.42]

        # --- goal ---
        gx, gy = to_px(c.goal)
        r = 4
        img[max(0, gy - r):min(img_h, gy + r + 1),
            max(0, gx - r):min(img_w, gx + r + 1)] = [0.18, 0.52, 0.18]

        # --- ants drawn FIRST so the T-shape renders on top ---
        ant_px_r = max(2, int(c.ant_radius / W * img_w))
        for p in self.ants:
            px, py = to_px(p)
            for dy in range(-ant_px_r, ant_px_r + 1):
                for dx in range(-ant_px_r, ant_px_r + 1):
                    if dx*dx + dy*dy <= ant_px_r*ant_px_r:
                        iy, ix = py + dy, px + dx
                        if 0 <= iy < img_h and 0 <= ix < img_w:
                            img[iy, ix] = [0.10, 0.25, 0.75]

        # --- T-shape on top (single solid fill, no outline) ---
        yy, xx = np.mgrid[0:img_h, 0:img_w]
        xw = xx / (img_w - 1) * W
        yw = (img_h - 1 - yy) / (img_h - 1) * H
        cos_a, sin_a = math.cos(self.obj.angle), math.sin(self.obj.angle)
        lx = cos_a * (xw - self.obj.center[0]) + sin_a * (yw - self.obj.center[1])
        ly = -sin_a * (xw - self.obj.center[0]) + cos_a * (yw - self.obj.center[1])
        mask = np.zeros((img_h, img_w), dtype=bool)
        for rect in self.obj.rects:
            rcx, rcy = rect.center
            rhx, rhy = rect.half_size
            mask |= (np.abs(lx - rcx) <= rhx) & (np.abs(ly - rcy) <= rhy)
        img[mask] = [0.67, 0.08, 0.055]

        return np.clip(img * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Demo GIF generator
# ---------------------------------------------------------------------------

def make_demo_gif(
    path: str | None = None,
    steps: int = 400,
    seed: int = 0,
    fps: int = 30,
    pose_seed: int | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out = Path(path) if path else Path(__file__).parent / "output" / "t_barrier_swarm.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = MovableTShape(pose_seed=pose_seed)
    env = AntSwarmEnv(cfg=cfg, seed=seed)
    env.reset(seed=seed)

    frames = []
    for s in range(steps):
        frames.append(env.render())
        _, _, terminated, truncated, info = env.step(env.heuristic_actions())
        if (s + 1) % 50 == 0:
            log.info(f"step {s+1}/{steps}  dist={info['object_distance']:.3f}")
        if terminated or truncated:
            break

    fig, ax = plt.subplots(figsize=(6.5, 6.5 * cfg.world_size[1] / cfg.world_size[0]))
    im = ax.imshow(frames[0])
    ax.set_axis_off()

    def update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.info(f"Saved → {out}  ({len(frames)} frames)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate T-shape ant swarm demo GIF")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pose-seed", type=int, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    result = make_demo_gif(
        path=args.out,
        steps=args.steps,
        seed=args.seed,
        fps=args.fps,
        pose_seed=args.pose_seed,
    )
    print(f"Saved: {result}")
