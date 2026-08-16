"""Reusable exploration controls for Stable-Baselines3 policies."""
from __future__ import annotations

import math
import warnings

import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class LogStdClampCallback(BaseCallback):
    """Bound a continuous policy's learned action log-standard-deviation.

    PPO updates happen between rollouts, so clamping at rollout start ensures
    every sampled action uses a bounded distribution. For gSDE policies, noise
    is resampled after clamping because SB3 initially samples it before invoking
    the rollout-start callback.

    Either bound may be ``None`` for a one-sided clamp. Passing ``None`` for
    both bounds is an error; callers should omit the callback instead.
    """

    def __init__(self, min_log_std: float | None = -3.0,
                 max_log_std: float | None = -0.5, verbose: int = 0):
        super().__init__(verbose)
        if min_log_std is None and max_log_std is None:
            raise ValueError("at least one log-std bound is required")
        if (min_log_std is not None and max_log_std is not None
                and min_log_std > max_log_std):
            raise ValueError("min_log_std must be <= max_log_std")
        self.min_log_std = None if min_log_std is None else float(min_log_std)
        self.max_log_std = None if max_log_std is None else float(max_log_std)
        self._missing_log_std_warned = False

    def _clamp(self, *, reset_sde_noise: bool) -> None:
        log_std = getattr(self.model.policy, "log_std", None)
        if log_std is None:
            if not self._missing_log_std_warned:
                warnings.warn(
                    "LogStdClampCallback: policy has no log_std parameter; "
                    "the clamp is inactive.", RuntimeWarning, stacklevel=2)
                self._missing_log_std_warned = True
            return

        with th.no_grad():
            below = ((log_std < self.min_log_std).sum().item()
                     if self.min_log_std is not None else 0)
            above = ((log_std > self.max_log_std).sum().item()
                     if self.max_log_std is not None else 0)
            log_std.clamp_(min=self.min_log_std, max=self.max_log_std)
            actual_min = float(log_std.min().item())
            actual_max = float(log_std.max().item())

        # collect_rollouts() samples gSDE weights before on_rollout_start().
        # Replace those weights so action noise reflects the clamped parameter.
        if reset_sde_noise and getattr(self.model, "use_sde", False):
            self.model.policy.reset_noise(self.training_env.num_envs)

        self.logger.record("exploration/log_std_min", actual_min)
        self.logger.record("exploration/log_std_max", actual_max)
        self.logger.record("exploration/std_min", math.exp(actual_min))
        self.logger.record("exploration/std_max", math.exp(actual_max))
        self.logger.record("exploration/log_std_values_clamped", below + above)

    def _on_training_start(self) -> None:
        self._clamp(reset_sde_noise=True)

    def _on_rollout_start(self) -> None:
        self._clamp(reset_sde_noise=True)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # A model saved immediately after learn() must be bounded as well.
        self._clamp(reset_sde_noise=False)
