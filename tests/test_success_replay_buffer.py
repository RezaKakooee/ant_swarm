from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rl"))

from success_replay_buffer import (  # noqa: E402
    SuccessReplayBuffer, seed_success_trajectories,
)
from train_sac import _settings, _validate_exact_resume  # noqa: E402


class SuccessReplayBufferTest(unittest.TestCase):
    def make_buffer(self, fraction=0.5):
        return SuccessReplayBuffer(
            buffer_size=64,
            observation_space=spaces.Box(-1000, 1000, shape=(2,), dtype=np.float32),
            action_space=spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
            device="cpu",
            n_envs=1,
            success_buffer_size=32,
            success_batch_fraction=fraction,
            reach_radius=0.1,
        )

    @staticmethod
    def add_step(buffer, value, *, done=False, success=False):
        obs = np.asarray([[value, value]], dtype=np.float32)
        next_obs = obs + 1
        action = np.asarray([[0.25]], dtype=np.float32)
        reward = np.asarray([float(success)], dtype=np.float32)
        infos = [{"is_success": success, "object_distance": 0.0 if success else 1.0}]
        buffer.add(obs, next_obs, action, reward, np.asarray([done]), infos)

    def test_only_complete_successful_episodes_are_retained(self):
        buffer = self.make_buffer()
        self.add_step(buffer, 0)
        self.add_step(buffer, 1, done=True, success=False)
        self.assertEqual(buffer.success_size, 0)

        self.add_step(buffer, 100)
        self.assertEqual(buffer.success_size, 0)
        self.add_step(buffer, 101)
        self.add_step(buffer, 102, done=True, success=True)
        self.assertEqual(buffer.success_size, 3)
        self.assertEqual(buffer.success_episodes, 1)
        self.assertEqual(buffer.success_transitions, 3)

    def test_sampling_reserves_success_fraction(self):
        buffer = self.make_buffer(fraction=0.5)
        for i in range(8):
            self.add_step(buffer, i, done=(i == 7), success=False)
        for i in range(4):
            self.add_step(buffer, 100 + i, done=(i == 3), success=(i == 3))

        sample = buffer.sample(20)
        # Ten rows are guaranteed to come from the success-only ring.  Uniform
        # samples from the main ring may add more successful rows.
        retained = (sample.observations[:, 0] >= 100).sum().item()
        self.assertGreaterEqual(retained, 10)
        self.assertEqual(sample.observations.shape[0], 20)

    def test_pending_episode_can_be_cleared_after_restore(self):
        buffer = self.make_buffer()
        self.add_step(buffer, 10)
        self.assertEqual(len(buffer._episode_transitions[0]), 1)
        buffer.clear_pending_episodes()
        self.assertEqual(buffer._episode_transitions, [[]])


    def test_target_entropy_is_read_from_config(self):
        cfg = OmegaConf.load(ROOT / "configs" / "rl" / "pnas_dyn_geo_v2.yaml")
        self.assertEqual(float(_settings(cfg)["target_entropy"]), -1.5)

    def test_exact_resume_rejects_reward_phase_change(self):
        source_cfg = OmegaConf.load(
            ROOT / "configs" / "rl" / "pnas_dyn_geo_v2.yaml"
        )
        target_cfg = OmegaConf.create(
            OmegaConf.to_container(source_cfg, resolve=True)
        )
        target_cfg.env.reward_mode = "sparse"

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            snapshot_dir = run_dir / "code"
            snapshot_dir.mkdir(parents=True)
            OmegaConf.save(source_cfg, snapshot_dir / "config.yaml")
            checkpoint = run_dir / "checkpoints" / "sac_final.zip"

            _validate_exact_resume(checkpoint, source_cfg)
            with self.assertRaisesRegex(ValueError, "phase/config mismatch"):
                _validate_exact_resume(checkpoint, target_cfg)


    def test_pickle_round_trip_restores_nested_success_buffer(self):
        buffer = self.make_buffer()
        self.add_step(buffer, 100)
        self.add_step(buffer, 101, done=True, success=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.pkl"
            save_to_pkl(path, buffer)
            restored = load_from_pkl(path)

        restored.set_device("cpu")
        restored.clear_pending_episodes()
        self.assertEqual(restored.size(), 2)
        self.assertEqual(restored.success_size, 2)
        self.assertEqual(restored.sample(4).observations.shape[0], 4)



    def test_json_seed_recomputes_sparse_reward(self):
        from ant_swarm import AntSwarmEnv

        cfg = OmegaConf.load(ROOT / "configs" / "rl" / "pnas_dyn_geo_v2.yaml")
        cfg.env.reward_mode = "sparse"
        probe = AntSwarmEnv(config=cfg, seed=0)
        # With angle zero, the tracked big-cap centre is stem_half left of the
        # object centre. Place that tracked point exactly on the goal; a zero
        # action then terminates with the sparse success bonus.
        center = [
            float(probe.layout.goal[0] + probe.tshape.stem_len / 2),
            float(probe.layout.goal[1]),
        ]
        trajectory = {
            "init_pose": [*center, 0.0],
            "actions": [[0.0, 0.0, 0.0]],
            "wall_len": float(probe.layout.wall_len),
        }
        replay = SuccessReplayBuffer(
            buffer_size=32,
            observation_space=flatten_space(probe.observation_space),
            action_space=probe.action_space,
            device="cpu",
            n_envs=1,
            success_buffer_size=16,
            success_batch_fraction=0.5,
            reach_radius=float(cfg.goal.reach_radius),
        )
        probe.close()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "success_one.json"
            path.write_text(json.dumps(trajectory))
            stats = seed_success_trajectories(replay, cfg, path)

        self.assertEqual(stats["loaded"], 1)
        self.assertEqual(stats["transitions"], 1)
        self.assertEqual(replay.size(), 1)
        self.assertEqual(replay.success_size, 1)
        self.assertAlmostEqual(
            float(replay.rewards[0, 0]), float(cfg.env.reward_success)
        )


if __name__ == "__main__":
    unittest.main()
