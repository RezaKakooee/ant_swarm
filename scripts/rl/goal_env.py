"""Goal-conditioned observation wrapper for HER (SB3 ``HerReplayBuffer``).

Turns the env's flat feature observation into the Dict layout HER expects:

    observation    — the usual flattened features
    achieved_goal  — where the goal-tracked point (big cap) actually is
    desired_goal   — where the episode's goal is

``compute_reward`` re-scores relabeled transitions: the sparse success bonus
if the achieved point is within the reach radius of the (relabeled) goal.
Relabeled goals may lie anywhere the load ever reached — left room and
corridor included — which is exactly what gives HER its implicit curriculum.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class GoalObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        u = env.unwrapped
        self._base_space = env.observation_space
        base = gym.spaces.flatten_space(env.observation_space)
        W, H = u.layout.world_size
        lim = float(max(W, H))
        goal_box = gym.spaces.Box(low=0.0, high=lim, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation": base,
            "achieved_goal": goal_box,
            "desired_goal": goal_box,
        })
        self.reward_success = float(u.cfg.env.reward_success)

    def _obs(self, obs):
        u = self.env.unwrapped
        return {
            "observation": np.asarray(
                gym.spaces.flatten(self._base_space, obs), dtype=np.float32),
            "achieved_goal": np.asarray(u.state.tracked_world(), dtype=np.float32),
            "desired_goal": np.asarray(u.layout.goal, dtype=np.float32),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs(obs), reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Sparse bonus for relabeled goals; batched by SB3."""
        d = np.linalg.norm(
            np.asarray(achieved_goal, dtype=np.float32)
            - np.asarray(desired_goal, dtype=np.float32), axis=-1)
        reach = float(self.env.unwrapped.layout.reach_radius)
        return (d < reach).astype(np.float32) * self.reward_success
