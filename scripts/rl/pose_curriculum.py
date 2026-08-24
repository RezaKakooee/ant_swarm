"""Exact-pose reverse curriculum built from a geodesic solution path.

The legacy reverse curriculum only moves an x-band while independently
sampling y and orientation.  That makes many starts unrelated to a valid
solution.  This module instead extracts one collision-free ``(x, y, theta)``
path from the precomputed geodesic field and walks fixed anchors from near the
goal back to the full task start.

Progression is mastery-only: a stage never advances because time elapsed.
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from ant_swarm.geodesic import PoseGeodesicField, curriculum_anchors
from ant_swarm.layout import Layout
from ant_swarm.tshape import TShape


def _cfg_value(section, *names, default=None):
    for name in names:
        value = getattr(section, name, None)
        if value is not None:
            return value
    return default


def _validate_anchors(cfg, wall_len: float, anchors: np.ndarray) -> None:
    """Fail before training if a field anchor is invalid in the real env."""
    path_cfg = copy.deepcopy(cfg)
    path_cfg.walls.length = float(wall_len)
    layout, tshape = Layout(path_cfg), TShape(path_cfg)
    track_local = tshape.track_local_point(getattr(path_cfg.env, "goal_track", "center"))
    W, H = layout.world_size
    for stage_idx, pose in enumerate(anchors):
        if not np.isfinite(pose).all():
            raise ValueError(f"pose-path anchor {stage_idx} is not finite: {pose.tolist()}")
        probe = tshape.clone_at(pose[:2], float(pose[2]))
        corners = probe.world_corners()
        in_world = (corners[:, 0].min() >= 0.0 and corners[:, 0].max() <= W
                    and corners[:, 1].min() >= 0.0 and corners[:, 1].max() <= H)
        if not in_world or probe.overlaps_walls(layout):
            raise ValueError(
                f"pose-path anchor {stage_idx} is not collision-free in the real env: "
                f"{pose.tolist()}"
            )
        if stage_idx == 0:
            tracked = probe.local_to_world(track_local)
            if np.linalg.norm(tracked - layout.goal) < layout.reach_radius:
                raise ValueError("pose-path stage 0 must begin outside the success radius")


class PosePathCurriculumCallback(BaseCallback):
    """Advance through exact solution-path poses only after stage mastery."""

    def __init__(self, anchors, *, success_threshold: float, window: int,
                 reach_radius: float, wall_len: float | None = None,
                 xy_jitter: float = 0.0, angle_jitter: float = 0.0,
                 start_stage: int = 0, stop_on_master: bool = False,
                 stop_success: float = 0.9, stop_window: int = 200,
                 final_spawn_x_range=None,
                 stall_log_steps: int | None = None, verbose: int = 1):
        super().__init__(verbose)
        anchors = np.asarray(anchors, dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) < 2:
            raise ValueError("pose-path anchors must have shape (N, 3), N >= 2")
        if not 0.0 <= float(success_threshold) <= 1.0:
            raise ValueError("success_threshold must be in [0, 1]")
        if int(window) < 1 or int(stop_window) < 1:
            raise ValueError("curriculum windows must be positive")
        if not 0 <= int(start_stage) < len(anchors):
            raise ValueError(f"start_stage must be in [0, {len(anchors) - 1}]")

        self.anchors = anchors
        # optional mirrored variant of every anchor (symmetric maze): each
        # episode randomly practises the up-turn or down-turn route
        self.mirror_anchors = None
        self.threshold = float(success_threshold)
        self.window = int(window)
        self.reach_radius = float(reach_radius)
        self.wall_len = None if wall_len is None else float(wall_len)
        self.xy_jitter = float(xy_jitter)
        self.angle_jitter = float(angle_jitter)
        self.stage_idx = int(start_stage)
        self.stop_on_master = bool(stop_on_master)
        self.stop_success = float(stop_success)
        # After the last anchor is mastered, switch to fully random spawns and
        # keep training (random-everything runs) instead of stopping.
        self.final_spawn_x_range = (None if final_spawn_x_range is None
                                    else (float(final_spawn_x_range[0]),
                                          float(final_spawn_x_range[1])))
        self._free_spawn = False
        self.success = deque(maxlen=self.window)
        self._target_success = deque(maxlen=int(stop_window))
        self._stage_start_step = 0
        self.stall_log_steps = (None if stall_log_steps is None
                                else max(int(stall_log_steps), 1))
        self._next_stall_log_step = None

    @property
    def at_target(self) -> bool:
        return self.stage_idx == len(self.anchors) - 1

    @property
    def current_pose(self) -> np.ndarray:
        return self.anchors[self.stage_idx]

    def _stage_poses(self, stage_idx: int):
        pose = [self.anchors[stage_idx].tolist()]
        if self.mirror_anchors is not None:
            pose.append(self.mirror_anchors[stage_idx].tolist())
        return pose

    def _set_pose(self, vec_env, stage_idx: int, *, exact: bool = False) -> None:
        xy_jitter = 0.0 if exact else self.xy_jitter
        angle_jitter = 0.0 if exact else self.angle_jitter
        vec_env.env_method(
            "set_spawn_pose", self._stage_poses(stage_idx), stage_idx,
            xy_jitter, angle_jitter)

    def prepare_env(self, vec_env, *, final: bool = False) -> None:
        """Pin layout and pose before SB3 performs its first reset."""
        if self.wall_len is not None:
            vec_env.env_method("set_wall_length", self.wall_len)
        stage_idx = len(self.anchors) - 1 if final else self.stage_idx
        self._set_pose(vec_env, stage_idx, exact=final)

    def prepare_raw_env(self, env, *, final: bool = False) -> None:
        """Raw-Gym equivalent of :meth:`prepare_env` for standalone eval."""
        if self.wall_len is not None:
            env.set_wall_length(self.wall_len)
        stage_idx = len(self.anchors) - 1 if final else self.stage_idx
        xy_jitter = 0.0 if final else self.xy_jitter
        angle_jitter = 0.0 if final else self.angle_jitter
        env.set_spawn_pose(self._stage_poses(stage_idx), stage_idx,
                           xy_jitter, angle_jitter)

    def _reset_stall_clock(self) -> None:
        self._next_stall_log_step = (
            None if self.stall_log_steps is None
            else self.num_timesteps + self.stall_log_steps
        )

    def _on_training_start(self) -> None:
        # Training scripts call prepare_env before model.learn(), because SB3
        # resets before this callback. Re-applying here keeps direct callback
        # users correct from their next episode onward.
        self._set_pose(self.training_env, self.stage_idx)
        self._stage_start_step = self.num_timesteps
        self._reset_stall_clock()
        if self.verbose:
            p = self.current_pose
            logger.info(
                f"[curriculum:pose_path] stage {self.stage_idx}/{len(self.anchors) - 1} "
                f"pose=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
            )

    def _advance(self, success_rate: float) -> None:
        previous = self.stage_idx
        self.stage_idx += 1
        self._set_pose(self.training_env, self.stage_idx)
        took = self.num_timesteps - self._stage_start_step
        self.logger.record("curriculum/stage_steps", took)
        if self.verbose:
            p = self.current_pose
            logger.info(
                f"[curriculum:pose_path] stage {previous} mastered in {took} steps "
                f"(success {success_rate:.2f}) -> stage {self.stage_idx} "
                f"pose=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
            )
        self.success.clear()
        self._stage_start_step = self.num_timesteps
        self._reset_stall_clock()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", ()),
                              self.locals.get("infos", ())):
            if not done:
                continue
            # Vec envs auto-reset before callbacks run. Completions from an
            # episode that began at an older stage must not master a new one.
            if int(info.get("curriculum_stage", -1)) != self.stage_idx:
                continue
            success = float(info.get(
                "is_success",
                info.get("object_distance", float("inf")) < self.reach_radius,
            ))
            self.success.append(success)
            if self.at_target:
                self._target_success.append(success)

        success_rate = (sum(self.success) / len(self.success)
                        if self.success else 0.0)
        if (not self.at_target and len(self.success) == self.window
                and success_rate >= self.threshold):
            self._advance(success_rate)

        if (self.at_target and not self._free_spawn
                and len(self._target_success) == self._target_success.maxlen):
            target_rate = sum(self._target_success) / len(self._target_success)
            if target_rate >= self.stop_success:
                if self.final_spawn_x_range is not None:
                    lo, hi = self.final_spawn_x_range
                    self.training_env.env_method("set_spawn_x_range", lo, hi)
                    self._free_spawn = True
                    logger.info(
                        f"[curriculum:pose_path] TARGET MASTERED (success "
                        f"{target_rate:.2f}) -> FREE SPAWN: random start in "
                        f"x=[{lo}, {hi}] (step {self.num_timesteps})."
                    )
                elif self.stop_on_master:
                    logger.info(
                        f"[curriculum:pose_path] TARGET MASTERED: success "
                        f"{target_rate:.2f} over {self._target_success.maxlen} episodes "
                        f"-> stopping training (step {self.num_timesteps})."
                    )
                    return False

        if (self._next_stall_log_step is not None
                and self.num_timesteps >= self._next_stall_log_step):
            logger.warning(
                f"[curriculum:pose_path] holding stage {self.stage_idx}: "
                f"success {success_rate:.2f} over {len(self.success)}/{self.window} "
                f"episodes after {self.num_timesteps - self._stage_start_step} steps"
            )
            elapsed = self.num_timesteps - self._next_stall_log_step
            self._next_stall_log_step += (elapsed // self.stall_log_steps + 1) * self.stall_log_steps

        pose = self.current_pose
        self.logger.record("curriculum/stage_idx", self.stage_idx)
        self.logger.record("curriculum/stage_fraction",
                           self.stage_idx / (len(self.anchors) - 1))
        self.logger.record("curriculum/anchor_x", float(pose[0]))
        self.logger.record("curriculum/anchor_y", float(pose[1]))
        self.logger.record("curriculum/anchor_theta", float(pose[2]))
        self.logger.record("curriculum/free_spawn", int(self._free_spawn))
        if self.success:
            self.logger.record("curriculum/success_rate", success_rate)
        return True


def build_pose_path_curriculum(cfg, cur, reach_radius: float) -> PosePathCurriculumCallback:
    """Build anchors and the mastery callback from ``curriculum:`` config."""
    field_path = _cfg_value(
        cur, "geodesic_field", default=getattr(cfg.env, "geodesic_field", None))
    if field_path is None:
        raise ValueError("pose_path curriculum requires env.geodesic_field")
    field = PoseGeodesicField(field_path)
    check_goal = str(getattr(cfg.env, "reward_mode", "")) != "geodesic_exit"
    field.validate_config(cfg, check_goal=check_goal)

    start_pose = getattr(cur, "path_start_pose", None)
    if start_pose is None:
        scale = float(cfg.scene_scale)
        x_range = [float(v) * scale for v in cfg.spawn.x_range]
        start_pose = field.farthest_pose(
            x_range=x_range,
            angle_range=tuple(map(float, cfg.spawn.angle_range)),
        )

    anchor_count = int(_cfg_value(
        cur, "path_anchor_count", "anchor_count", "path_stages", default=16))
    anchors = curriculum_anchors(
        field, start_pose, anchor_count,
        near_goal_fraction=float(getattr(cur, "near_goal_fraction", 0.05)),
    )
    wall_len = _cfg_value(
        cur, "path_wall_len", "reverse_wall_len", default=cfg.walls.length)
    _validate_anchors(cfg, float(wall_len), anchors)

    callback = PosePathCurriculumCallback(
        anchors,
        success_threshold=float(cur.success_threshold),
        window=int(cur.window),
        reach_radius=reach_radius,
        wall_len=float(wall_len),
        xy_jitter=float(getattr(cur, "xy_jitter", 0.0)),
        angle_jitter=float(getattr(cur, "angle_jitter", 0.0)),
        start_stage=int(getattr(cur, "start_stage", 0)),
        stop_on_master=bool(getattr(cur, "stop_on_master", False)),
        stop_success=float(getattr(cur, "stop_success", 0.9)),
        stop_window=int(getattr(cur, "stop_window", 200)),
        final_spawn_x_range=getattr(cur, "final_spawn_x_range", None),
        stall_log_steps=getattr(cur, "stall_log_steps", None),
    )
    if str(getattr(cur, "route", "down")) == "up":
        # train ONLY the up-turn route: replace every anchor by its mirror
        H = float(cfg.world.height) * float(cfg.scene_scale)
        up = anchors.copy()
        up[:, 1] = H - up[:, 1]
        up[:, 2] = -up[:, 2]
        _validate_anchors(cfg, float(wall_len), up)
        callback.anchors = up
        logger.info("[curriculum:pose_path] route=up: all anchors mirrored; "
                    "the down-route path is never practised")
    if bool(getattr(cur, "mirror_anchors", False)):
        # y -> H - y, theta -> -theta: valid because walls, goal height and the
        # T-shape are all symmetric about the maze's mid-height
        H = float(cfg.world.height) * float(cfg.scene_scale)
        mirrored = anchors.copy()
        mirrored[:, 1] = H - mirrored[:, 1]
        mirrored[:, 2] = -mirrored[:, 2]
        _validate_anchors(cfg, float(wall_len), mirrored)
        callback.mirror_anchors = mirrored
        logger.info("[curriculum:pose_path] mirrored anchors ON: each episode "
                    "randomly uses the up-turn or down-turn route")
    if getattr(cur, "max_steps_per_stage", None) is not None:
        logger.info("[curriculum:pose_path] max_steps_per_stage ignored; progression is mastery-only")
    return callback
