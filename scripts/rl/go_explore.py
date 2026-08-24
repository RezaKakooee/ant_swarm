"""Go-Explore, phase 1, as a config-switchable module. Does NOT train a policy.

Idea (Ecoffet et al. 2019): keep an ARCHIVE of visited cells (discretised
load poses). Repeat: pick an archive cell (novelty-weighted), RETURN to it
exactly (the simulator is deterministic: re-apply the stored actions from the
episode start), then EXPLORE with random actions for a while. New cells enter
the archive with the action trail that reaches them.

Restore-by-replay keeps this exact without a state-snapshot API: an entry
stores (spawn_pose, goal, actions); resetting with those overrides and
replaying the actions reproduces the state bit-for-bit.

Every goal hit is written as a success JSON in the SAME format the training
callback saves (`train_utils.SuccessTrajectoryCallback`), so phase 2 is the
machinery we already have: `run.seed_successes_from` + the success replay
buffer, or a pose-path curriculum built from the found trajectory.

Config (``go_explore:`` section; see run_go_explore.py):
    cell_xy: 0.02        # archive grid, metres
    cell_deg: 30         # archive grid, degrees
    explore_steps: 60    # random steps per exploration burst
    select_power: 1.0    # cell weight = 1 / (visits + 1) ** power
    max_len: 450         # drop trails longer than this (episode cap is 500)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class GoExplore:
    def __init__(self, env, *, cell_xy=0.02, cell_deg=30.0, explore_steps=60,
                 select_power=1.0, max_len=450, rng=None):
        self.env = env
        self.u = env.unwrapped if hasattr(env, "unwrapped") else env
        self.cell_xy = float(cell_xy)
        self.cell_th = math.radians(float(cell_deg))
        self.explore_steps = int(explore_steps)
        self.select_power = float(select_power)
        self.max_len = int(max_len)
        self.rng = rng or np.random.default_rng(0)
        # remember the env's own sampling setup so fresh episodes use it
        self._orig_goal_choices = self.u._goal_choices
        self._orig_resample = self.u._resample_each_reset
        # cell -> dict(spawn_pose, goal, actions, visits)
        self.archive: dict = {}
        self.successes: list = []
        self.steps_used = 0

    # ------------------------------------------------------------------
    def _cell(self):
        c = self.u.state.object_center
        th = float(self.u.state.object_angle) % (2 * math.pi)
        return (int(float(c[0]) / self.cell_xy),
                int(float(c[1]) / self.cell_xy),
                int(th / self.cell_th))

    def _restore(self, entry):
        """Reset to the entry's episode start and replay its action trail."""
        self.u.set_goal_choices([entry["goal"]])
        self.u.set_spawn_pose([entry["spawn_pose"]], 0, 0.0, 0.0)
        self.env.reset(seed=0)
        for a in entry["actions"]:
            self.env.step(np.asarray(a, dtype=np.float32).reshape(1, -1))
        self.steps_used += len(entry["actions"])

    def _select(self):
        cells = list(self.archive)
        w = np.array([1.0 / (self.archive[c]["visits"] + 1.0) ** self.select_power
                      for c in cells])
        return self.archive[cells[self.rng.choice(len(cells), p=w / w.sum())]]

    def _fresh_episode(self):
        """Start a brand-new episode; the env samples spawn/goal as configured."""
        self.u._spawn_pose_override = None            # clear the restore pins
        self.u._goal_choices = self._orig_goal_choices
        self.u._resample_each_reset = self._orig_resample
        self.env.reset(seed=int(self.rng.integers(1 << 31)))
        return {"spawn_pose": [float(self.u.init_center[0]), float(self.u.init_center[1]),
                               float(self.u.init_angle)],
                "goal": [float(v) for v in self.u.layout.goal],
                "actions": [], "visits": 0}

    # ------------------------------------------------------------------
    def run(self, total_steps: int, log_every: int = 50_000, log=print):
        """Explore until ``total_steps`` env steps are spent. Returns stats."""
        next_log = log_every
        while self.steps_used < total_steps:
            if self.archive and self.rng.random() < 0.95:
                entry = dict(self._select())          # copy: the trail will grow
                entry["actions"] = list(entry["actions"])
                self._restore(entry)
            else:
                entry = self._fresh_episode()
            src = self.archive.get(self._cell())
            if src is not None:
                src["visits"] += 1
            done = False
            for _ in range(self.explore_steps):
                a = self.rng.uniform(-1.0, 1.0, size=self.u.action_space.shape) \
                        .astype(np.float32)
                _, _, term, trunc, info = self.env.step(a.reshape(1, -1))
                entry["actions"].append(a.ravel().tolist())
                self.steps_used += 1
                if info["is_success"]:
                    self.successes.append({k: entry[k] for k in
                                           ("spawn_pose", "goal", "actions")})
                    done = True
                    break
                if term or trunc or len(entry["actions"]) >= self.max_len:
                    done = True
                    break
                cell = self._cell()
                known = self.archive.get(cell)
                if known is None or len(entry["actions"]) < len(known["actions"]):
                    self.archive[cell] = {"spawn_pose": entry["spawn_pose"],
                                          "goal": entry["goal"],
                                          "actions": list(entry["actions"]),
                                          "visits": 0}
            if self.steps_used >= next_log:
                next_log += log_every
                log(f"[go-explore] steps={self.steps_used}  cells={len(self.archive)}  "
                    f"successes={len(self.successes)}  frontier_x={self.frontier_x():.3f}")
            if done:
                continue
        return self.stats()

    # ------------------------------------------------------------------
    def frontier_x(self) -> float:
        """How far right the archive reaches (progress through the maze)."""
        return max((c[0] for c in self.archive), default=0) * self.cell_xy

    def stats(self) -> dict:
        return {"steps": self.steps_used, "cells": len(self.archive),
                "successes": len(self.successes), "frontier_x": self.frontier_x()}

    def save_successes(self, out_dir) -> int:
        """Write found solutions in the training callback's success format."""
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(self.successes):
            payload = {"init_pose": s["spawn_pose"], "actions": s["actions"],
                       "wall_len": float(self.u.layout.wall_len),
                       "gap": float(self.u.layout.gap),
                       "curriculum_stage": -1, "goal": s["goal"],
                       "length": len(s["actions"]), "episode_return": 1.0,
                       "final_distance": 0.0, "timesteps": self.steps_used,
                       "source": "go_explore"}
            f = out / f"success_goexplore_{i:04d}_len{len(s['actions'])}.json"
            json.dump(payload, open(f, "w"))
        return len(self.successes)
