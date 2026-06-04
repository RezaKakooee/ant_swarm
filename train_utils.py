"""Shared training-side callbacks (SB3-dependent — kept out of the env package).

The env package (`ant_swarm`) stays RL-library-agnostic; anything importing
stable_baselines3 lives here.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SuccessTrajectoryCallback(BaseCallback):
    """Persist every SUCCESSFUL episode's trajectory to disk.

    Successful episodes are rare and valuable on this hard-exploration task, so we
    save them for later reuse (behaviour cloning, demonstrations, warm-starting,
    HER-style relabelling). We store the **minimal set to deterministically replay**
    the episode (the env has no randomness in ``step``), not the full obs/rewards::

        init_pose      [cx, cy, angle]    where the T started (world coords)
        actions        list[T][act_dim]   the action sequence (env received)
        wall_len, gap                     curriculum stage / layout
        length, episode_return, final_distance, timesteps   (metadata)

    obs/rewards/path are recovered by replaying actions from ``init_pose`` at
    ``wall_len`` (see ``render_success.replay_obs`` / ``_frames_for``) — ~10x
    smaller files. obs is still collected in-memory here, only for the dedup
    signature and to read the initial pose.

    Works for both on-policy (PPO) and off-policy (SAC): at callback time
    ``model._last_obs`` is still the pre-step observation, and ``locals`` holds the
    action/reward/done/info for the step just taken.

    Dedup: with a fixed spawn pose a converged policy repeats nearly the same path
    every episode, producing huge numbers of near-identical successes. When
    ``dedup=True`` (default) we hash a downsampled, rounded version of the object
    *path* (smooth, ~deterministic) and skip trajectories whose signature was
    already saved — keeping only distinct solutions.
    """

    def __init__(self, save_dir, reach_radius: float, max_keep: int | None = None,
                 dedup: bool = True, sig_points: int = 16, dedup_tol: float = 0.05, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.reach_radius = reach_radius
        self.max_keep = max_keep
        self.dedup = dedup
        self.sig_points = sig_points     # waypoints sampled along the path
        self.dedup_tol = dedup_tol        # max-abs path diff below which two trajs are "the same"
        self._traj = None
        self._n_saved = 0
        self._n_dup = 0
        self._kept_sigs = []              # downsampled path summaries of saved trajectories

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        n = self.training_env.num_envs
        self._traj = [{"obs": [], "act": [], "rew": []} for _ in range(n)]
        # world size (to denormalise obs → world init pose); obs is collected only
        # in-memory for dedup + init-pose, never persisted.
        try:
            self._W, self._H = self.training_env.get_attr("layout")[0].world_size
        except Exception:
            from ant_swarm import load_config
            c = load_config(); s = float(c.scene_scale)
            self._W, self._H = c.world.width * s, c.world.height * s

    def _path_summary(self, tr):
        """Downsampled object PATH (obs cols obj_x,obj_y,goal_dx,dy,sin,cos).

        Uses the smooth, ~deterministic object trajectory (not noisy actions),
        sampled at `sig_points` waypoints.
        """
        obs = np.asarray(tr["obs"], dtype=np.float32)       # (T, obs_dim)
        path = obs[:, 2:8]
        n = len(path)
        idx = np.linspace(0, n - 1, min(n, self.sig_points)).astype(int)
        return path[idx]                                    # (P, 6)

    def _is_duplicate(self, summary) -> bool:
        """True if `summary` is within `dedup_tol` of an already-saved trajectory.

        Tolerance-based (not hashing) so small action noise doesn't create false
        distinctness. The kept set stays small (distinct solutions only), so the
        scan is cheap.
        """
        for k in self._kept_sigs:
            if k.shape == summary.shape and np.max(np.abs(k - summary)) < self.dedup_tol:
                return True
        return False

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
                    if self.dedup:
                        summary = self._path_summary(tr)
                        if self._is_duplicate(summary):
                            self._n_dup += 1
                        else:
                            self._kept_sigs.append(summary)
                            self._save(tr, infos[i])
                    else:
                        self._save(tr, infos[i])
                self._traj[i] = {"obs": [], "act": [], "rew": []}
        return True

    def _save(self, tr, info):
        length = len(tr["act"])
        ret = float(np.sum(tr["rew"]))
        # initial pose from the first (pre-step) obs: obj_x,obj_y (cols 2,3) + sin,cos (6,7)
        o0 = np.asarray(tr["obs"][0], dtype=float)
        init_pose = [float(o0[2]) * self._W, float(o0[3]) * self._H,
                     math.atan2(float(o0[6]), float(o0[7]))]
        fname = self.save_dir / f"success_t{self.num_timesteps:09d}_ep{self._n_saved:04d}_len{length}.json"
        payload = {
            # minimal & sufficient to deterministically replay the episode:
            "init_pose": init_pose,                          # [cx, cy, angle] (world)
            "actions": [np.asarray(a, dtype=float).ravel().tolist() for a in tr["act"]],
            "wall_len": float(info.get("wall_len", float("nan"))),  # curriculum stage / layout
            "gap": float(info.get("gap", float("nan"))),
            # metadata (derivable, kept for convenience/filtering):
            "length": length,
            "episode_return": ret,
            "final_distance": float(info.get("object_distance", float("nan"))),
            "timesteps": int(self.num_timesteps),
        }
        with open(fname, "w") as f:
            json.dump(payload, f)
        self._n_saved += 1
        if self.verbose:
            print(f"  [success] saved {fname.name}  len={length}  return={ret:.3f}  "
                  f"(unique={self._n_saved}, dup-skipped={self._n_dup})", flush=True)
        try:
            self.logger.record("rollout/successes_saved", self._n_saved)
            self.logger.record("rollout/successes_dup_skipped", self._n_dup)
        except Exception:
            pass
        self._prune()

    def _prune(self):
        if not self.max_keep:
            return
        files = sorted(self.save_dir.glob("success_*.json"))
        for f in files[:-self.max_keep]:
            try:
                f.unlink()
            except OSError:
                pass


class CurriculumCallback(BaseCallback):
    """Difficulty curriculum that advances when rolling success is high enough.

    Two modes (``mode``):

    * ``gap``     — anneal the barrier **wall length** start→target (wider→narrower
      gap). Difficulty applied via ``env.set_wall_length``.
    * ``reverse`` — pin the gap at ``wall_len_pin`` and anneal the **spawn x**
      start→target (T starts past the barrier near the goal, then moves backward
      to the full spawn). Difficulty applied via ``env.set_spawn_x_range`` (a
      ±``band`` window). Reward stays the true (sparse) reward — this only changes
      *where episodes begin*, giving the agent winnable starts so it ever sees the
      success signal; it still learns the maneuver itself.

    Stall safety: a stage that doesn't reach ``success_threshold`` within
    ``max_steps_per_stage`` is force-advanced (``None`` to disable).
    """

    def __init__(self, start, target, step, success_threshold, window,
                 reach_radius, max_steps_per_stage=2_000_000,
                 stop_on_master=False, stop_success=0.9, stop_window=200,
                 mode="gap", band=0.05, wall_len_pin=None, verbose=1):
        super().__init__(verbose)
        self.mode = mode
        self.band = float(band)
        self.wall_len_pin = wall_len_pin
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

    def _set_difficulty(self, value):
        if self.mode == "reverse":
            self.training_env.env_method("set_spawn_x_range", value - self.band, value + self.band)
        else:
            self.training_env.env_method("set_wall_length", value)

    def _do_advance(self, reason: str):
        if self.target >= self.start:
            self.current = min(self.current + self.step, self.target)
        else:
            self.current = max(self.current - self.step, self.target)
        self._set_difficulty(self.current)
        took = self.num_timesteps - self._stage_start_step
        self.logger.record("curriculum/stage_steps", took)
        self.logger.record("curriculum/stage_idx", self._stage_idx)
        if self.verbose:
            knob = "spawn_x" if self.mode == "reverse" else "wall_len"
            print(f"[curriculum] stage {self._stage_idx} done in {took} steps "
                  f"({reason}) → {knob}={self.current:.3f}", flush=True)
        self._stage_idx += 1
        self._stage_start_step = self.num_timesteps
        self._eps_since_advance = 0
        self.success.clear()

    def _on_training_start(self) -> None:
        if self.mode == "reverse" and self.wall_len_pin is not None:
            self.training_env.env_method("set_wall_length", float(self.wall_len_pin))
        self._set_difficulty(self.current)
        self._stage_start_step = self.num_timesteps
        if self.verbose:
            knob = "spawn_x" if self.mode == "reverse" else "wall_len"
            print(f"[curriculum:{self.mode}] start {knob}={self.current:.3f} "
                  f"(target {self.target:.3f})", flush=True)

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
            knob = "spawn_x" if self.mode == "reverse" else "wall_len"
            print(f"[curriculum] TARGET MASTERED: success {sr:.2f} over "
                  f"{self._target_success.maxlen} episodes at {knob}={self.current:.3f} "
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

        self.logger.record("curriculum/difficulty", self.current)   # wall_len or spawn_x
        if self.success:
            self.logger.record("curriculum/success_rate", sum(self.success) / len(self.success))
        return True


def build_curriculum(cur, reach_radius) -> CurriculumCallback:
    """Construct a CurriculumCallback from the config's `curriculum:` section."""
    mode = getattr(cur, "mode", "gap")
    common = dict(
        success_threshold=cur.success_threshold, window=cur.window, reach_radius=reach_radius,
        max_steps_per_stage=getattr(cur, "max_steps_per_stage", None),
        stop_on_master=getattr(cur, "stop_on_master", False),
        stop_success=getattr(cur, "stop_success", 0.9),
        stop_window=getattr(cur, "stop_window", 200),
        mode=mode,
    )
    if mode == "reverse":
        return CurriculumCallback(
            start=cur.start_spawn_x, target=cur.target_spawn_x, step=cur.spawn_step,
            band=getattr(cur, "spawn_band", 0.05), wall_len_pin=cur.reverse_wall_len, **common)
    return CurriculumCallback(
        start=cur.start_wall_len, target=cur.target_wall_len, step=cur.step, **common)


def pin_eval_hard(eval_env, cur):
    """Pin the eval env at the real hard task (so eval reflects true difficulty)."""
    mode = getattr(cur, "mode", "gap")
    if mode == "reverse":
        band = getattr(cur, "spawn_band", 0.05)
        eval_env.env_method("set_wall_length", float(cur.reverse_wall_len))
        eval_env.env_method("set_spawn_x_range", cur.target_spawn_x - band, cur.target_spawn_x + band)
    else:
        eval_env.env_method("set_wall_length", float(cur.target_wall_len))
