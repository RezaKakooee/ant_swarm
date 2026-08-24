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
        # --- generalisation options (all optional, default = old behaviour) ---
        spawn_cfg = self.cfg.spawn
        fixed = getattr(spawn_cfg, "fixed_pose", None)
        if fixed is not None:               # always start from this exact pose
            self.init_center = np.array(fixed[:2], dtype=np.float32)
            self.init_angle = float(fixed[2])
        goals = getattr(self.cfg.goal, "random_positions", None)
        self._goal_choices = (np.array(goals, dtype=np.float32) * float(self.cfg.scene_scale)
                              if goals else None)
        box = getattr(self.cfg.goal, "random_box", None)   # [[xmin,ymin],[xmax,ymax]]
        self._goal_box = (np.array(box, dtype=np.float32) * float(self.cfg.scene_scale)
                          if box else None)

        self._prev_dist = 0.0
        self._pending_wall_len = None       # gap curriculum: applied on next reset
        self._spawn_x_override = None       # legacy reverse curriculum: spawn x-band
        self._spawn_pose_override = None    # pose-path curriculum: [x, y, theta]
        self._spawn_pose_jitter = (0.0, 0.0)
        self._curriculum_stage = -1
        self._episode_curriculum_stage = -1
        # fresh pose every episode (random-start training, or a curriculum)
        self._resample_each_reset = bool(getattr(spawn_cfg, "resample_each_reset", False))
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
        """Legacy reverse curriculum: sample x, y and theta independently."""
        self._spawn_x_override = (float(lo), float(hi))
        self._spawn_pose_override = None
        self._curriculum_stage = -1
        self._resample_each_reset = True

    def set_spawn_pose(self, pose, stage_idx=-1, xy_jitter=0.0, angle_jitter=0.0):
        """Curriculum anchor(s): one ``(x, y, theta)`` pose, or several — one
        row is drawn at random each reset (e.g. a path and its mirror)."""
        p = np.asarray(pose, dtype=np.float32).reshape(-1, 3)
        self._spawn_pose_override = p
        self._spawn_pose_jitter = (float(xy_jitter), float(angle_jitter))
        self._curriculum_stage = int(stage_idx)
        self._spawn_x_override = None
        self._resample_each_reset = True
        self._sample_anchor_pose()  # validate immediately, before training starts

    def set_goal_choices(self, goals):
        """Goal curriculum: restrict which goals episodes may draw from.

        Stage 0 is usually just the goal the policy already knows; later stages
        add the others. Takes effect on the next reset.
        """
        self._goal_choices = (np.array(goals, dtype=np.float32)
                              * float(self.cfg.scene_scale)) if goals is not None else None

    def get_goal_choices(self):
        return None if self._goal_choices is None else self._goal_choices.tolist()

    def set_reward_mode(self, mode):
        """Switch reward phase in place (used for geodesic -> sparse tuning)."""
        self.cfg.env.reward_mode = str(mode)
        self.reward_model = RewardModel(self.cfg)
        if self.state.obj is not None:
            self.reward_model.reset(self.state)

    def _pose_is_free(self, center, angle):
        probe = self.tshape.clone_at(center, angle)
        c = probe.world_corners()
        W, H = self.layout.world_size
        return bool(c[:, 0].min() >= 0.0 and c[:, 0].max() <= W
                    and c[:, 1].min() >= 0.0 and c[:, 1].max() <= H
                    and not probe.overlaps_walls(self.layout))

    def _sample_anchor_pose(self):
        anchors = self._spawn_pose_override
        xy_jitter, angle_jitter = self._spawn_pose_jitter
        for k in self.rng.permutation(len(anchors)):   # random variant per episode
            anchor = anchors[k]
            for _ in range(100):
                center = anchor[:2] + self.rng.uniform(-xy_jitter, xy_jitter, size=2)
                angle = float(anchor[2] + self.rng.uniform(-angle_jitter, angle_jitter))
                if self._pose_is_free(center, angle):
                    return np.asarray(center, dtype=np.float32), angle
                if xy_jitter == 0.0 and angle_jitter == 0.0:
                    break
        raise ValueError(
            f"no collision-free curriculum anchor in: {anchors.tolist()}")

    def _sample_spawn(self):
        if self._spawn_pose_override is not None:
            return self._sample_anchor_pose()
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

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._apply_pending()
        goal = None
        if self._goal_choices is not None:             # random goal from a list
            goal = self._goal_choices[self.rng.integers(len(self._goal_choices))]
        elif self._goal_box is not None:               # continuous random goal
            goal = self.rng.uniform(self._goal_box[0], self._goal_box[1]) \
                       .astype(np.float32)
        if goal is not None:
            self.layout.goal[:] = goal                 # obs model holds this array by reference
            self.reward_model.set_goal(goal)
        if self._resample_each_reset:                  # fresh start each episode
            self.init_center, self.init_angle = self._sample_spawn()
        self._episode_curriculum_stage = self._curriculum_stage
        self.state.reset(self.init_center, self.init_angle)
        self._prev_dist = self.state.distance_to_goal()
        self.reward_model.reset(self.state)      # geodesic mode re-anchors its potential
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
        reward, reached = self.reward_model.compute(
            dist, self._prev_dist, self.layout.reach_radius, state=self.state)
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
            "is_success": bool(reached),
            "curriculum_stage": self._episode_curriculum_stage,
            "goal": [float(self.layout.goal[0]), float(self.layout.goal[1])],
            "spawn_pose": [float(self.init_center[0]), float(self.init_center[1]),
                           float(self.init_angle)],
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
