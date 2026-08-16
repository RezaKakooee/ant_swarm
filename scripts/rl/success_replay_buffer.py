"""Replay helpers for retaining rare successful SAC episodes.

The ordinary SB3 replay buffer eventually overwrites rare successes.  This
buffer keeps the normal replay distribution, but also stores complete
successful episodes in a second ring and reserves part of every minibatch for
those transitions.

Saved success JSON files can be replayed through the *current* environment and
inserted into the buffer.  Replaying is important when moving from geodesic
pretraining to sparse fine-tuning: observations and sparse rewards are rebuilt
instead of reusing stale geodesic rewards.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


Transition = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]


class SuccessReplayBuffer(ReplayBuffer):
    """Uniform replay plus an episode-level success archive.

    A transition is copied to the success ring only after its episode has
    finished successfully.  This avoids incorrectly retaining promising-looking
    prefixes of episodes that ultimately fail.
    """

    def __init__(
        self,
        *args,
        success_buffer_size: int = 200_000,
        success_batch_fraction: float = 0.25,
        reach_radius: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if success_buffer_size <= 0:
            raise ValueError("success_buffer_size must be positive")
        if not 0.0 <= success_batch_fraction <= 1.0:
            raise ValueError("success_batch_fraction must be in [0, 1]")

        self.success_batch_fraction = float(success_batch_fraction)
        self.reach_radius = None if reach_radius is None else float(reach_radius)
        self.success_buffer = ReplayBuffer(
            buffer_size=int(success_buffer_size),
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=self.handle_timeout_termination,
        )
        self._episode_transitions: list[list[Transition]] = [
            [] for _ in range(self.n_envs)
        ]
        self.success_episodes = 0
        self.success_transitions = 0

    @property
    def success_size(self) -> int:
        """Number of transitions currently retained in the success ring."""
        return self.success_buffer.size()

    def _transition_for_env(
        self,
        env_idx: int,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: dict[str, Any],
    ) -> Transition:
        action = np.asarray(action).reshape((self.n_envs, self.action_dim))
        reward = np.asarray(reward).reshape(self.n_envs)
        done = np.asarray(done).reshape(self.n_envs)
        return (
            np.asarray(obs[env_idx : env_idx + 1]).copy(),
            np.asarray(next_obs[env_idx : env_idx + 1]).copy(),
            action[env_idx : env_idx + 1].copy(),
            reward[env_idx : env_idx + 1].copy(),
            done[env_idx : env_idx + 1].copy(),
            [dict(info)],
        )

    def _is_success(self, info: dict[str, Any]) -> bool:
        if "is_success" in info:
            return bool(info["is_success"])
        if self.reach_radius is not None and "object_distance" in info:
            return float(info["object_distance"]) < self.reach_radius
        return False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add normal replay and retain a completed successful episode."""
        obs_arr = np.asarray(obs)
        next_obs_arr = np.asarray(next_obs)
        done_arr = np.asarray(done).reshape(self.n_envs)

        # The main ring always sees every transition.
        super().add(obs, next_obs, action, reward, done, infos)

        for env_idx in range(self.n_envs):
            transition = self._transition_for_env(
                env_idx, obs_arr, next_obs_arr, action, reward, done_arr, infos[env_idx]
            )
            episode = self._episode_transitions[env_idx]
            episode.append(transition)
            if not bool(done_arr[env_idx]):
                continue

            if self._is_success(infos[env_idx]):
                for item in episode:
                    self.success_buffer.add(*item)
                self.success_episodes += 1
                self.success_transitions += len(episode)
            self._episode_transitions[env_idx] = []

    @staticmethod
    def _combine_samples(
        regular: ReplayBufferSamples,
        successful: ReplayBufferSamples,
    ) -> ReplayBufferSamples:
        values = []
        for field in ReplayBufferSamples._fields:
            left = getattr(regular, field)
            right = getattr(successful, field)
            if left is None and right is None:
                values.append(None)
            elif left is None or right is None:
                raise ValueError(f"incompatible replay samples for field {field}")
            else:
                values.append(th.cat((left, right), dim=0))

        size = values[0].shape[0]
        order = th.randperm(size, device=values[0].device)
        values = [value if value is None else value[order] for value in values]
        return ReplayBufferSamples(*values)

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        success_size = self.success_buffer.size()
        regular_size = self.size()
        if batch_size <= 1 and success_size and regular_size:
            if self.success_batch_fraction >= 0.5:
                return self.success_buffer.sample(batch_size, env=env)
            return super().sample(batch_size, env=env)
        if success_size == 0 or self.success_batch_fraction == 0.0:
            return super().sample(batch_size, env=env)
        if regular_size == 0 or self.success_batch_fraction == 1.0:
            return self.success_buffer.sample(batch_size, env=env)

        n_success = int(round(batch_size * self.success_batch_fraction))
        n_success = min(max(n_success, 1), batch_size - 1)
        regular = super().sample(batch_size - n_success, env=env)
        successful = self.success_buffer.sample(n_success, env=env)
        return self._combine_samples(regular, successful)

    def clear_pending_episodes(self) -> None:
        """Drop partial episodes after restoring replay into reset environments."""
        self._episode_transitions = [[] for _ in range(self.n_envs)]

    def set_device(self, device) -> None:
        """Update both buffers after SB3 unpickles replay on a new device."""
        self.device = device
        self.success_buffer.device = device


def _json_paths(source: str | Path) -> list[Path]:
    source = Path(source).expanduser()
    if source.is_dir():
        return sorted(source.rglob("success_*.json"))
    if source.is_file():
        return [source]
    # A glob is useful for selecting only one curriculum stage/run.
    return sorted(source.parent.glob(source.name))


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def seed_success_trajectories(
    replay_buffer: SuccessReplayBuffer,
    cfg,
    source: str | Path,
    *,
    max_trajectories: int | None = None,
) -> dict[str, int]:
    """Replay saved successes under ``cfg`` and seed both replay rings.

    Rewards are recomputed by the current environment.  Consequently a
    geodesic demonstration imported into a sparse run contains only the current
    sparse terminal bonus, never its old shaping rewards.
    """
    if replay_buffer.n_envs != 1:
        raise ValueError("success JSON seeding currently requires a one-env replay buffer")

    # Local imports keep this standalone buffer module usable without importing
    # the project environment during pickle loading.
    from gymnasium.spaces.utils import flatten

    from ant_swarm import AntSwarmEnv

    paths = _json_paths(source)
    stats = {"loaded": 0, "transitions": 0, "skipped": 0, "files": len(paths)}
    env = AntSwarmEnv(config=cfg, seed=0)
    target_wall_len = float(env.layout.wall_len)

    for path in paths:
        try:
            trajectories = _load_trajectories(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            stats["skipped"] += 1
            continue

        for trajectory in trajectories:
            if max_trajectories is not None and stats["loaded"] >= max_trajectories:
                env.close()
                return stats
            actions = trajectory.get("actions")
            init_pose = trajectory.get("init_pose")
            if not actions or not isinstance(init_pose, (list, tuple)) or len(init_pose) != 3:
                stats["skipped"] += 1
                continue

            wall_len = trajectory.get("wall_len")
            if wall_len is not None:
                try:
                    wall_len = float(wall_len)
                except (TypeError, ValueError):
                    stats["skipped"] += 1
                    continue
                if np.isfinite(wall_len) and not np.isclose(
                    wall_len, target_wall_len, rtol=0.0, atol=1e-6
                ):
                    stats["skipped"] += 1
                    continue

            env.reset()
            center = np.asarray(init_pose[:2], dtype=np.float32)
            angle = float(init_pose[2])
            env.state.reset(center, angle)
            if env.state._bad_pose():
                stats["skipped"] += 1
                continue
            env.init_center, env.init_angle = center.copy(), angle
            env._prev_dist = env.state.distance_to_goal()
            env.reward_model.reset(env.state)
            obs = flatten(env.observation_space, env.obs_model.observe(env.state)).astype(np.float32)

            episode: list[Transition] = []
            succeeded = False
            for raw_action in actions:
                try:
                    action = np.asarray(raw_action, dtype=np.float32).reshape(env.action_space.shape)
                except (TypeError, ValueError):
                    episode = []
                    break
                next_raw, reward, terminated, truncated, info = env.step(action)
                next_obs = flatten(env.observation_space, next_raw).astype(np.float32)
                done = bool(terminated or truncated)
                item_info = dict(info)
                item_info["TimeLimit.truncated"] = bool(truncated and not terminated)
                episode.append((
                    obs[None, :],
                    next_obs[None, :],
                    action.reshape(1, -1),
                    np.asarray([reward], dtype=np.float32),
                    np.asarray([done], dtype=bool),
                    [item_info],
                ))
                obs = next_obs
                if done:
                    succeeded = bool(info.get("is_success", terminated))
                    break

            if not succeeded or not episode:
                stats["skipped"] += 1
                continue

            # The episode is already verified, so adding it cannot leave a
            # failed prefix in either ring.  The final info triggers archival.
            for item in episode:
                replay_buffer.add(*item)
            stats["loaded"] += 1
            stats["transitions"] += len(episode)

    env.close()
    return stats


__all__ = ["SuccessReplayBuffer", "seed_success_trajectories"]
