"""``AntSwarmEnv`` — gym env composing the layout + state + action/obs/reward.

N ants are rigidly attached to a T-shaped object and must push it past barrier
walls to a goal.  This module wires together the modular components:

    config  → Layout, TShape
            → ObservationModel, ActionModel, RewardModel, SwarmState
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._gym import gym
from .config import load_config
from .layout import Layout
from .tshape import TShape, sample_free_pose, make_attachment_offsets
from .state import SwarmState
from .render import Renderer
from .action import ActionModel
from .observation import ObservationModel
from .reward import RewardModel


class AntSwarmEnv(gym.Env):
    """Gym/Gymnasium env: ants threading a T-shape through a barrier to a goal."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config=None, *, max_steps=None, render_mode="rgb_array",
                 seed=None, ant_offsets=None):
        super().__init__()
        self.cfg = config if (config is not None and not isinstance(config, (str, Path))) \
            else load_config(config)
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        # static scene + object template
        self.layout = Layout(self.cfg)
        self.tshape = TShape(self.cfg)

        # modular components
        self.action_model = ActionModel(self.cfg)
        self.obs_model = ObservationModel(self.cfg, self.layout, self.tshape)
        self.reward_model = RewardModel(self.cfg)

        self.n_ants = int(self.cfg.ants.n)
        self.ant_radius = self.cfg.ants.radius * float(self.cfg.scene_scale)
        self.attachment_offsets = make_attachment_offsets(
            self.tshape, self.n_ants, self.rng, ant_offsets)

        self.state = SwarmState(self.cfg, self.layout, self.tshape, self.attachment_offsets)

        self.max_steps = int(max_steps if max_steps is not None else self.cfg.env.max_steps)

        self.action_space = self.action_model.space()
        self.observation_space = self.obs_model.space()

        # fixed spawn pose for this env instance
        self.init_center, self.init_angle = sample_free_pose(
            self.tshape, self.layout, self.rng,
            x_range=self.cfg.spawn.x_range,
            angle_range=self.cfg.spawn.angle_range,
            margin=self.cfg.spawn.margin,
            max_tries=int(self.cfg.spawn.max_tries),
        )
        self._prev_dist = 0.0
        self._pending_wall_len = None       # gap curriculum: applied on next reset
        self._spawn_x_override = None       # reverse curriculum: spawn x-band
        self._resample_each_reset = False   # reverse curriculum: fresh pose each episode
        self.renderer = Renderer(self.cfg, self.layout)

    # ------------------------------------------------------------------
    # Curriculum hook
    # ------------------------------------------------------------------
    def set_wall_length(self, value):
        """Schedule a new barrier wall length (curriculum); takes effect next reset."""
        self._pending_wall_len = float(value)

    def get_wall_length(self):
        return self.layout.wall_len

    def set_spawn_x_range(self, lo, hi):
        """Reverse curriculum: spawn the T in x-band [lo, hi] (config units),
        resampling a fresh pose each episode. Takes effect on next reset."""
        self._spawn_x_override = (float(lo), float(hi))
        self._resample_each_reset = True

    def _sample_spawn(self):
        x_range = self._spawn_x_override or self.cfg.spawn.x_range
        return sample_free_pose(
            self.tshape, self.layout, self.rng,
            x_range=x_range, angle_range=self.cfg.spawn.angle_range,
            margin=self.cfg.spawn.margin, max_tries=int(self.cfg.spawn.max_tries))

    def _apply_pending(self):
        if self._pending_wall_len is None:
            return
        self.cfg.walls.length = self._pending_wall_len
        self.layout = Layout(self.cfg)                                   # walls + heads + gap
        self.obs_model = ObservationModel(self.cfg, self.layout, self.tshape)  # refresh wall_heads
        self.state = SwarmState(self.cfg, self.layout, self.tshape, self.attachment_offsets)
        self.renderer = Renderer(self.cfg, self.layout)
        self.init_center, self.init_angle = self._sample_spawn()
        self._pending_wall_len = None
        if self._pending_wall_len is None:
            return
        self.cfg.walls.length = self._pending_wall_len
        self.layout = Layout(self.cfg)                                   # walls + heads + gap
        self.obs_model = ObservationModel(self.cfg, self.layout, self.tshape)  # refresh wall_heads
        self.state = SwarmState(self.cfg, self.layout, self.tshape, self.attachment_offsets)
        self.renderer = Renderer(self.cfg, self.layout)
        self.init_center, self.init_angle = sample_free_pose(
            self.tshape, self.layout, self.rng,
            x_range=self.cfg.spawn.x_range, angle_range=self.cfg.spawn.angle_range,
            margin=self.cfg.spawn.margin, max_tries=int(self.cfg.spawn.max_tries))
        self._pending_wall_len = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._apply_pending()
        if self._resample_each_reset:                  # reverse curriculum: fresh start each episode
            self.init_center, self.init_angle = self._sample_spawn()
        self.state.reset(self.init_center, self.init_angle)
        self._prev_dist = self.state.distance_to_goal()
        return self.obs_model.observe(self.state), {}

    def step(self, actions):
        self.state.step_count += 1

        if self.action_model.mode == "kinematic":
            direction, rotation = self.action_model.decode_kinematic(actions)
            self.state.apply_kinematic(direction, rotation,
                                       self.action_model.step_len, self.action_model.rot_step)
        else:
            force, torque = self.action_model.to_wrench(actions, self.state)
            self.state.integrate(force, torque)

        dist = self.state.distance_to_goal()
        reward, reached = self.reward_model.compute(dist, self._prev_dist, self.layout.reach_radius)
        self._prev_dist = dist

        terminated = bool(reached)
        truncated = self.state.step_count >= self.max_steps
        info = {
            "object_center": self.state.object_center.copy(),
            "object_angle": self.state.object_angle,
            "object_distance": dist,
            "step": self.state.step_count,
            "wall_len": self.layout.wall_len,   # current difficulty (curriculum)
            "gap": self.layout.gap,
        }
        return self.obs_model.observe(self.state), reward, terminated, truncated, info

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def close(self):
        pass

    # convenience for callers/scripts
    @property
    def ants(self):
        return self.state.ants

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        return self.renderer.render(self.state)


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


def make_compat_env(**kwargs):
    return GymCompatWrapper(AntSwarmEnv(**kwargs))
