"""Reward component.

Two modes (config ``env.reward_mode``):

  * ``shaped`` (default): dense distance shaping + sparse success bonus::
        reward = progress_coef * (prev_dist - dist) + (success if reached)
  * ``sparse``: only the success bonus::
        reward = success if reached else 0

``reached`` = the goal-tracking point is within the goal's reach radius.
"""
from __future__ import annotations


class RewardModel:
    def __init__(self, cfg):
        self.mode = getattr(cfg.env, "reward_mode", "shaped")
        self.progress_coef = cfg.env.reward_progress_coef
        self.success = cfg.env.reward_success

    def compute(self, dist: float, prev_dist: float, reach_radius: float):
        """Return ``(reward, reached)`` for one transition."""
        reached = dist < reach_radius
        bonus = self.success if reached else 0.0
        if self.mode == "sparse":
            return bonus, reached
        return self.progress_coef * (prev_dist - dist) + bonus, reached
