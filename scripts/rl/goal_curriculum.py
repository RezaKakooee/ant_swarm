"""Goal curriculum: widen the set of goals as the agent keeps succeeding.

The policy learns the maneuver toward one goal first, then has to generalise to
more distant ones. Same principle that solved the original task: never jump
straight to the hardest distribution.

A stage is passed only on real mastery (rolling success >= threshold over a
window of episodes); time alone never promotes it.
"""
from __future__ import annotations

from collections import deque

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class GoalCurriculumCallback(BaseCallback):
    def __init__(self, stages, *, success_threshold: float = 0.7, window: int = 100,
                 reach_radius: float = 0.06, verbose: int = 1):
        super().__init__(verbose)
        self.stages = [list(s) for s in stages]      # stage -> list of goals
        self.threshold = float(success_threshold)
        self.window = int(window)
        self.reach_radius = float(reach_radius)
        self.idx = 0
        self.success = deque(maxlen=self.window)
        self._episodes_at_stage = 0

    # -- helpers ------------------------------------------------------
    def _apply(self, vec_env) -> None:
        vec_env.env_method("set_goal_choices", self.stages[self.idx])

    def prepare_env(self, vec_env, *, final: bool = False) -> None:
        """Pin an eval env at the full goal set (or the current stage)."""
        vec_env.env_method("set_goal_choices",
                           self.stages[-1] if final else self.stages[self.idx])

    # -- callback -----------------------------------------------------
    def _on_training_start(self) -> None:
        self._apply(self.training_env)
        logger.info(f"[goal curriculum] stage 0/{len(self.stages)-1}: "
                    f"{len(self.stages[0])} goal(s) {self.stages[0]}")

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.success.append(float(info.get("object_distance", 1.0) < self.reach_radius))
            self._episodes_at_stage += 1

        if self.idx < len(self.stages) - 1 and len(self.success) == self.window \
                and self._episodes_at_stage >= self.window:
            rate = sum(self.success) / len(self.success)
            if rate >= self.threshold:
                self.idx += 1
                self._apply(self.training_env)
                self.success.clear()
                self._episodes_at_stage = 0
                logger.info(f"[goal curriculum] stage {self.idx}/{len(self.stages)-1} "
                            f"(success {rate:.2f}) -> {len(self.stages[self.idx])} goals")

        self.logger.record("curriculum/goal_stage", self.idx)
        self.logger.record("curriculum/goal_count", len(self.stages[self.idx]))
        if self.success:
            self.logger.record("curriculum/goal_success",
                               sum(self.success) / len(self.success))
        return True
