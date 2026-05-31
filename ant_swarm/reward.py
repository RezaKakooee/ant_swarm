"""Reward component.

Dense shaping toward the goal plus a sparse success bonus::

    reward = progress_coef * (prev_dist - dist) + (success if reached else 0)

where ``reached`` means the T-shape centre is within the goal's reach radius.
"""
from __future__ import annotations


class RewardModel:
    def __init__(self, cfg):
        self.progress_coef = cfg.env.reward_progress_coef
        self.success = cfg.env.reward_success

    def compute(self, dist: float, prev_dist: float, reach_radius: float):
        """Return ``(reward, reached)`` for one transition."""
        progress = prev_dist - dist
        reached = dist < reach_radius
        reward = self.progress_coef * progress + (self.success if reached else 0.0)
        return reward, reached
