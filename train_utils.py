"""Shared training-side callbacks (SB3-dependent — kept out of the env package).

The env package (`ant_swarm`) stays RL-library-agnostic; anything importing
stable_baselines3 lives here.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SuccessTrajectoryCallback(BaseCallback):
    """Persist every SUCCESSFUL episode's trajectory to disk.

    Successful episodes are rare and valuable on this hard-exploration task, so we
    save them for later reuse (behaviour cloning, demonstrations, warm-starting,
    HER-style relabelling). Each is a compressed ``.npz`` with::

        obs            (T, obs_dim)   the policy inputs (pre-step, flattened)
        actions        (T, act_dim)   the actions the env received
        rewards        (T,)
        length, episode_return, final_distance, timesteps

    Works for both on-policy (PPO) and off-policy (SAC): at callback time
    ``model._last_obs`` is still the pre-step observation, and ``locals`` holds the
    action/reward/done/info for the step just taken.
    """

    def __init__(self, save_dir, reach_radius: float, max_keep: int | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.reach_radius = reach_radius
        self.max_keep = max_keep
        self._traj = None
        self._n_saved = 0

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        n = self.training_env.num_envs
        self._traj = [{"obs": [], "act": [], "rew": []} for _ in range(n)]

    def _on_step(self) -> bool:
        obs = self.model._last_obs                                   # pre-step obs
        actions = self.locals.get("clipped_actions", self.locals["actions"])
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for i in range(len(dones)):
            tr = self._traj[i]
            tr["obs"].append(np.asarray(obs[i], dtype=np.float32))
            tr["act"].append(np.asarray(actions[i], dtype=np.float32))
            tr["rew"].append(float(rewards[i]))
            if dones[i]:
                if infos[i].get("object_distance", 1.0) < self.reach_radius:
                    self._save(tr, infos[i])
                self._traj[i] = {"obs": [], "act": [], "rew": []}
        return True

    def _save(self, tr, info):
        length = len(tr["act"])
        ret = float(np.sum(tr["rew"]))
        fname = self.save_dir / f"success_t{self.num_timesteps:09d}_ep{self._n_saved:04d}_len{length}.npz"
        np.savez_compressed(
            fname,
            obs=np.asarray(tr["obs"], dtype=np.float32),
            actions=np.asarray(tr["act"], dtype=np.float32),
            rewards=np.asarray(tr["rew"], dtype=np.float32),
            length=length,
            episode_return=ret,
            final_distance=float(info.get("object_distance", float("nan"))),
            timesteps=int(self.num_timesteps),          # training step it was found at
            wall_len=float(info.get("wall_len", float("nan"))),  # curriculum difficulty
            gap=float(info.get("gap", float("nan"))),
        )
        self._n_saved += 1
        if self.verbose:
            print(f"  [success] saved {fname.name}  len={length}  return={ret:.3f}", flush=True)
        try:
            self.logger.record("rollout/successes_saved", self._n_saved)
        except Exception:
            pass
        self._prune()

    def _prune(self):
        if not self.max_keep:
            return
        files = sorted(self.save_dir.glob("success_*.npz"))
        for f in files[:-self.max_keep]:
            try:
                f.unlink()
            except OSError:
                pass


class CurriculumCallback(BaseCallback):
    """Gap-size curriculum: widen→narrow the barrier as the agent succeeds.

    Starts the (training) barrier at an easy ``start`` wall length and steps it
    toward the hard ``target`` whenever the rolling success rate over the last
    ``window`` episodes reaches ``success_threshold``. Difficulty is changed via
    each env's ``set_wall_length`` (applied on the next episode reset).

    Stall safety: if a stage hasn't advanced after ``max_steps_per_stage`` env
    steps it **force-advances** anyway, so a too-hard stage can't park the
    curriculum forever (which would waste the rest of training at a sub-target
    gap). Set ``max_steps_per_stage=None`` to disable.

    (Larger wall length = narrower gap = harder. Works in either direction.)
    """

    def __init__(self, start, target, step, success_threshold, window,
                 reach_radius, max_steps_per_stage=2_000_000,
                 stop_on_master=False, stop_success=0.9, stop_window=200, verbose=1):
        super().__init__(verbose)
        self.start = float(start)
        self.target = float(target)
        self.step = abs(float(step))
        self.threshold = success_threshold
        self.window = window
        self.reach_radius = reach_radius
        self.max_steps_per_stage = max_steps_per_stage
        self.stop_on_master = stop_on_master
        self.stop_success = stop_success
        self.current = self.start
        self.success = deque(maxlen=window)
        self._target_success = deque(maxlen=stop_window)   # success at the final target
        self._eps_since_advance = 0
        self._stage_start_step = 0       # num_timesteps when this stage began
        self._stage_idx = 0

    def _at_target(self) -> bool:
        return abs(self.current - self.target) < 1e-9

    def _do_advance(self, reason: str):
        if self.target >= self.start:
            self.current = min(self.current + self.step, self.target)
        else:
            self.current = max(self.current - self.step, self.target)
        self.training_env.env_method("set_wall_length", self.current)
        took = self.num_timesteps - self._stage_start_step
        self.logger.record("curriculum/stage_steps", took)
        self.logger.record("curriculum/stage_idx", self._stage_idx)
        if self.verbose:
            print(f"[curriculum] stage {self._stage_idx} done in {took} steps "
                  f"({reason}) → wall_len={self.current:.3f}", flush=True)
        self._stage_idx += 1
        self._stage_start_step = self.num_timesteps
        self._eps_since_advance = 0
        self.success.clear()

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_wall_length", self.current)
        self._stage_start_step = self.num_timesteps
        if self.verbose:
            print(f"[curriculum] start wall_len={self.current:.3f} (target {self.target:.3f})", flush=True)

    def _on_step(self) -> bool:
        at_target = self._at_target()
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                succ = float(info.get("object_distance", 1.0) < self.reach_radius)
                self.success.append(succ)
                self._eps_since_advance += 1
                if at_target:
                    self._target_success.append(succ)

        # early stop: target mastered over a sustained window
        if (self.stop_on_master and at_target
                and len(self._target_success) == self._target_success.maxlen
                and sum(self._target_success) / len(self._target_success) >= self.stop_success):
            sr = sum(self._target_success) / len(self._target_success)
            print(f"[curriculum] TARGET MASTERED: success {sr:.2f} over "
                  f"{self._target_success.maxlen} episodes at wall_len={self.current:.3f} "
                  f"→ stopping training (step {self.num_timesteps}).", flush=True)
            return False   # stops model.learn()

        if not self._at_target():
            sr = (sum(self.success) / len(self.success)) if self.success else 0.0
            # normal advance: enough episodes at high enough success
            if len(self.success) >= self.window and self._eps_since_advance >= self.window \
                    and sr >= self.threshold:
                self._do_advance(f"success {sr:.2f}")
            # stall safety: stage took too long → force advance
            elif (self.max_steps_per_stage is not None
                  and self.num_timesteps - self._stage_start_step >= self.max_steps_per_stage):
                self._do_advance(f"stall>{self.max_steps_per_stage} steps, sr={sr:.2f}")

        self.logger.record("curriculum/wall_len", self.current)
        if self.success:
            self.logger.record("curriculum/success_rate", sum(self.success) / len(self.success))
        return True
