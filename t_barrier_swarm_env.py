"""Ant swarm pushing a T-shaped object through barrier gates — gym env.

Modular layout:
  * ``config.yaml``  — single source of truth for all parameters
  * ``swarm_config`` — loads the YAML into an attribute namespace
  * ``geometry``     — LocalRect + SAT collision helpers
  * ``layout``       — world outline + inner barrier walls + goal
  * ``tshape``       — the movable T-shaped object
  * this file        — the gym ``AntSwarmEnv`` that composes the above

The environment is OpenAI-Gym / Gymnasium compatible and registered as
``"AntSwarmBarrier-v0"``.

Usage:
    python t_barrier_swarm_env.py            # saves a random-policy GIF
    import gymnasium as gym; gym.make("AntSwarmBarrier-v0")
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# --- gym / gymnasium dual-import shim ---
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_PACKAGE = "gymnasium"
except ImportError:
    import gym  # type: ignore[no-redef]
    from gym import spaces  # type: ignore[no-redef]
    _GYM_PACKAGE = "gym"

from loguru import logger as log
from rich.logging import RichHandler

from swarm_config import load_config
from layout import Layout
from tshape import TShape, sample_free_pose

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class AntSwarmEnv(gym.Env):
    """N ants rigidly attached to a T-shape, pushing it past barrier walls.

    observation_space : Box(-1, 1, shape=(n_ants, 9))
        Per ant: [att_local_x, att_local_y, obj_x, obj_y, goal_dx, goal_dy,
                  sin_angle, cos_angle, ang_vel].
    action_space : Box(shape=(n_ants, 2))
        Per ant radial force [angle ∈ [-π, π], magnitude ∈ [0, 1]].
    reward : float
        ``reward_progress_coef * progress + reward_success`` on reaching goal.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config=None,
        *,
        max_steps: int | None = None,
        render_mode: str = "rgb_array",
        seed=None,
        ant_offsets: np.ndarray | None = None,
    ):
        super().__init__()
        # config may be a namespace, a path, or None (→ default config.yaml)
        self.cfg = config if (config is not None and not isinstance(config, (str, Path))) \
            else load_config(config)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        s = float(self.cfg.scene_scale)
        self.layout = Layout(self.cfg)
        self.tshape = TShape(self.cfg)        # template; self.obj is a posed clone

        # physics params (with scene scaling, matching original behaviour)
        ph = self.cfg.physics
        self.push_strength = ph.push_strength * s
        self.object_mass = ph.object_mass
        self.object_inertia = ph.object_inertia * s * s
        self.linear_friction = ph.linear_friction
        self.angular_friction = ph.angular_friction
        self.substeps = int(ph.substeps)
        self.boundary_margin = ph.boundary_margin * s
        self.rest_wall = ph.restitution_wall
        self.rest_bound = ph.restitution_bound

        self.n_ants = int(self.cfg.ants.n)
        self.ant_radius = self.cfg.ants.radius * s

        self.max_steps = int(max_steps if max_steps is not None else self.cfg.env.max_steps)
        self.reward_progress_coef = self.cfg.env.reward_progress_coef
        self.reward_success = self.cfg.env.reward_success

        n = self.n_ants
        self.action_space = spaces.Box(
            low=np.array([[-math.pi, 0.0]] * n, dtype=np.float32),
            high=np.array([[math.pi, 1.0]] * n, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n, 9), dtype=np.float32)

        self.attachment_offsets = self._make_attachment_offsets(ant_offsets)

        # sample a (fixed) spawn pose for this env instance
        self.init_center, self.init_angle = sample_free_pose(
            self.tshape, self.layout, self.rng,
            x_range=self.cfg.spawn.x_range,
            angle_range=self.cfg.spawn.angle_range,
            margin=self.cfg.spawn.margin,
            max_tries=int(self.cfg.spawn.max_tries),
        )

        self.obj: TShape | None = None
        self.ants: np.ndarray | None = None
        self.step_count = 0
        self._prev_dist = 0.0

    # ------------------------------------------------------------------
    # Attachment offsets
    # ------------------------------------------------------------------
    def _make_attachment_offsets(self, ant_offsets) -> np.ndarray:
        if ant_offsets is not None:
            return np.array(ant_offsets, dtype=np.float32)
        if self.n_ants == 2:
            # one ant at each T-junction (stem ↔ cap), in local frame
            return np.array([[-self.tshape.stem_len / 2, 0.0],
                             [ self.tshape.stem_len / 2, 0.0]], dtype=np.float32)
        # otherwise sample uniformly on the T perimeter
        rects = self.tshape.rects
        perims = [4.0 * (float(r.half_size[0]) + float(r.half_size[1])) for r in rects]
        total = sum(perims)
        probs = [p / total for p in perims]
        offsets = []
        for _ in range(self.n_ants):
            r_idx = int(self.rng.choice(len(rects), p=probs))
            rect = rects[r_idx]
            cx, cy = float(rect.center[0]), float(rect.center[1])
            hx, hy = float(rect.half_size[0]), float(rect.half_size[1])
            perim = 2 * (2 * hx + 2 * hy)
            t = self.rng.uniform(0, perim)
            if t < 2 * hx:
                x, y = cx - hx + t, cy - hy
            elif t < 2 * hx + 2 * hy:
                x, y = cx + hx, cy - hy + (t - 2 * hx)
            elif t < 4 * hx + 2 * hy:
                x, y = cx + hx - (t - 2 * hx - 2 * hy), cy + hy
            else:
                x, y = cx - hx, cy + hy - (t - 4 * hx - 2 * hy)
            offsets.append([x, y])
        return np.array(offsets, dtype=np.float32)

    def _update_ant_positions(self):
        self.ants = np.array(
            [self.obj.local_to_world(self.attachment_offsets[i]) for i in range(self.n_ants)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.obj = self.tshape.clone_at(self.init_center, self.init_angle)
        self._prev_dist = float(np.linalg.norm(self.obj.center - self.layout.goal))
        self._update_ant_positions()
        return self._obs(), {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)  # (n_ants, 2): [angle, magnitude]
        self.step_count += 1

        total_force = np.zeros(2, dtype=np.float32)
        total_torque = 0.0
        for i in range(self.n_ants):
            angle = float(actions[i, 0])
            magnitude = float(np.clip(actions[i, 1], 0.0, 1.0))
            if magnitude < 1e-8:
                continue
            move_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            force = self.push_strength * magnitude * move_dir
            attachment_world = self.obj.local_to_world(self.attachment_offsets[i])
            arm = attachment_world - self.obj.center
            total_force += force
            total_torque += arm[0] * force[1] - arm[1] * force[0]

        self._integrate_object(total_force, total_torque)
        self._update_ant_positions()

        dist = float(np.linalg.norm(self.obj.center - self.layout.goal))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        reached = dist < self.layout.reach_radius
        reward = self.reward_progress_coef * progress + (self.reward_success if reached else 0.0)

        terminated = bool(reached)
        truncated = self.step_count >= self.max_steps
        info = {
            "object_center": self.obj.center.copy(),
            "object_angle": self.obj.angle,
            "object_distance": dist,
            "step": self.step_count,
        }
        return self._obs(), reward, terminated, truncated, info

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _integrate_object(self, force, torque):
        self.obj.vel = self.linear_friction * self.obj.vel + force / self.object_mass
        self.obj.ang_vel = self.angular_friction * self.obj.ang_vel + torque / self.object_inertia

        n_sub = self.substeps
        rw = self.rest_wall
        dp = self.obj.vel / n_sub
        da = self.obj.ang_vel / n_sub
        for _ in range(n_sub):
            prev_c = self.obj.center.copy()
            prev_a = self.obj.angle
            self.obj.center = prev_c + dp
            self.obj.angle = prev_a + da
            if self.obj.overlaps_walls(self.layout):
                self.obj.center = np.array([prev_c[0] + dp[0], prev_c[1]], dtype=np.float32)
                if self.obj.overlaps_walls(self.layout):
                    self.obj.center = np.array([prev_c[0], prev_c[1] + dp[1]], dtype=np.float32)
                    if self.obj.overlaps_walls(self.layout):
                        self.obj.center = prev_c
                        self.obj.angle = prev_a
                        self.obj.vel *= rw
                        self.obj.ang_vel *= rw
                        break
                    else:
                        self.obj.vel[0] *= rw
                else:
                    self.obj.vel[1] *= rw
            if self.obj.overlaps_walls(self.layout):
                self.obj.angle = prev_a
                self.obj.ang_vel *= rw

        W, H = self.layout.world_size
        corners = self.obj.world_corners()
        shift = np.zeros(2, dtype=np.float32)
        m = self.boundary_margin
        rb = self.rest_bound
        if corners[:, 0].min() < m:
            shift[0] += m - corners[:, 0].min(); self.obj.vel[0] *= rb
        if corners[:, 0].max() > W - m:
            shift[0] -= corners[:, 0].max() - (W - m); self.obj.vel[0] *= rb
        if corners[:, 1].min() < m:
            shift[1] += m - corners[:, 1].min(); self.obj.vel[1] *= rb
        if corners[:, 1].max() > H - m:
            shift[1] -= corners[:, 1].max() - (H - m); self.obj.vel[1] *= rb
        self.obj.center += shift

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _obs(self):
        W, H = self.layout.world_size
        stem_half = self.tshape.stem_len / 2
        cap_half = max(self.tshape.cap_big_len, self.tshape.cap_small_len) / 2
        obs = np.zeros((self.n_ants, 9), dtype=np.float32)
        goal_d = (self.layout.goal - self.obj.center) / np.array([W, H], dtype=np.float32)
        for i in range(self.n_ants):
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
    # Rendering (pure NumPy)
    # ------------------------------------------------------------------
    def render(self):
        lay = self.layout
        W, H = lay.world_size
        img_w, img_h = 650, int(650 * H / W)
        img = np.full((img_h, img_w, 3), 0.72, dtype=np.float32)

        def to_px(xy):
            x, y = xy
            return int(x / W * (img_w - 1)), int((H - y) / H * (img_h - 1))

        # walls: extend outward (away from passage) so the gap stays open
        half_H = H / 2
        ht_w = lay.wall_thickness / 2
        hl = lay.wall_len / 2
        extra = lay.wall_render_extra
        for wx, wy in lay.wall_positions:
            if wy > half_H:
                y_inner, y_outer = wy - hl, wy + hl + extra
            else:
                y_inner, y_outer = wy + hl, wy - hl - extra
            y_lo, y_hi = min(y_inner, y_outer), max(y_inner, y_outer)
            x0, y0 = to_px((wx - ht_w, y_lo))
            x1, y1 = to_px((wx + ht_w, y_hi))
            xs0, xs1 = max(0, min(x0, x1)), min(img_w, max(x0, x1) + 1)
            ys0, ys1 = max(0, min(y0, y1)), min(img_h, max(y0, y1) + 1)
            img[ys0:ys1, xs0:xs1] = [0.38, 0.38, 0.42]

        # goal
        gx, gy = to_px(lay.goal)
        r = 4
        img[max(0, gy - r):min(img_h, gy + r + 1),
            max(0, gx - r):min(img_w, gx + r + 1)] = [0.18, 0.52, 0.18]

        # T-shape
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

        # ants as bright white circles on top
        ant_px_r = max(4, int(self.ant_radius / W * img_w * 2))
        for p in self.ants:
            px, py = to_px(p)
            for dy in range(-ant_px_r, ant_px_r + 1):
                for dx in range(-ant_px_r, ant_px_r + 1):
                    if dx * dx + dy * dy <= ant_px_r * ant_px_r:
                        iy, ix = py + dy, px + dx
                        if 0 <= iy < img_h and 0 <= ix < img_w:
                            img[iy, ix] = [1.0, 1.0, 1.0]

        return np.clip(img * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Classic-gym compatibility wrapper
# ---------------------------------------------------------------------------
class GymCompatWrapper(gym.Wrapper):
    """Classic OpenAI Gym 4-tuple API: ``obs = reset()``, ``(obs,rew,done,info)``."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info["terminated"] = terminated
        info["truncated"] = truncated
        return obs, reward, done, info

    def seed(self, seed=None):
        return self.env.seed(seed)


# ---------------------------------------------------------------------------
# Gym registration
# ---------------------------------------------------------------------------
def _register():
    try:
        ids = {spec.id for spec in gym.envs.registry.values()}
    except Exception:
        ids = set()
    _max = int(load_config().env.max_steps)
    if "AntSwarmBarrier-v0" not in ids:
        gym.register(id="AntSwarmBarrier-v0",
                     entry_point=f"{__name__}:AntSwarmEnv", max_episode_steps=_max)
    if "AntSwarmBarrier-v0-compat" not in ids:
        gym.register(id="AntSwarmBarrier-v0-compat",
                     entry_point=f"{__name__}:_make_compat_env", max_episode_steps=_max)


def _make_compat_env(**kwargs):
    return GymCompatWrapper(AntSwarmEnv(**kwargs))


_register()


# ---------------------------------------------------------------------------
# Demo GIF generator
# ---------------------------------------------------------------------------
def make_demo_gif(path=None, steps=400, seed=0, fps=30) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out = Path(path) if path else Path(__file__).parent / "output" / "t_barrier_swarm.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    env = AntSwarmEnv(seed=seed)
    env.reset(seed=seed)

    frames = []
    for s in range(steps):
        frames.append(env.render())
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if (s + 1) % 50 == 0:
            log.info(f"step {s+1}/{steps}  dist={info['object_distance']:.3f}")
        if terminated or truncated:
            break

    W, H = env.layout.world_size
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * H / W))
    im = ax.imshow(frames[0]); ax.set_axis_off()
    anim = FuncAnimation(fig, lambda i: [im.set_data(frames[i]) or im],
                         frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    log.info(f"Saved → {out}  ({len(frames)} frames)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate T-shape ant swarm demo GIF")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    result = make_demo_gif(path=args.out, steps=args.steps, seed=args.seed, fps=args.fps)
    print(f"Saved: {result}")
