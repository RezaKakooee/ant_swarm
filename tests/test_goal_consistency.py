from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ant_swarm import AntSwarmEnv, load_config
from scripts.il.repair_replay_goals import audit_goals, goal_vectors, require_correct_goals


class GoalConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / 'configs/il/il_augmented_bc.yaml')
        self.env = AntSwarmEnv(config=self.cfg, seed=0)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def inferred_goal(self):
        obs = self.env.obs_model.observe(self.env.state)[0]
        return self.env.state.tracked_world() + obs[4:6] * self.env.layout.world_size

    def test_reassign_goal_updates_observation_reward_and_does_not_alias_dataset(self):
        dataset_goals = np.array([[1.4, .2], [1.3, .5]], dtype=np.float32)
        original = dataset_goals.copy()
        for goal in dataset_goals:
            self.env.layout.goal = goal
            np.testing.assert_allclose(self.inferred_goal(), goal, atol=2e-7)
            self.env.reset()
        np.testing.assert_array_equal(dataset_goals, original)
        # The compatibility setter must also update the task, not detach obs.
        self.env.obs_model.goal = self.env.state.tracked_world()
        _, reward, term, _, info = self.env.step(np.zeros((1, 3), dtype=np.float32))
        self.assertTrue(term)
        self.assertTrue(info['is_success'])
        self.assertEqual(reward, self.cfg.env.reward_success)

    def test_explicit_reset_is_independent_of_rng_and_episode_history(self):
        options = {'init_pose': [0.35, 0.36, 0.0], 'goal': [1.3, .5]}
        action = np.array([[.2, .7, -.3]], dtype=np.float32)
        trajectories = []
        for seed in (1, 987):
            obs, _ = self.env.reset(seed=seed, options=options)
            self.assertEqual(self.env.state.step_count, 0)
            self.assertEqual(self.env._prev_dist, self.env.state.distance_to_goal())
            trajectory = [obs.copy()]
            for _ in range(8):
                trajectory.append(self.env.step(action)[0].copy())
            trajectories.append(trajectory)
        np.testing.assert_array_equal(trajectories[0], trajectories[1])
        np.testing.assert_allclose(self.inferred_goal(), options['goal'], atol=2e-7)

    def test_repair_matches_fresh_observations_for_all_tracking_points_and_scales(self):
        for scale in (1.0, 2.0):
            for tracking in ('center', 'big_cap', 'small_cap'):
                cfg = self.cfg.copy()
                cfg.scene_scale = scale
                cfg.env.goal_track = tracking
                env = AntSwarmEnv(config=cfg, seed=0)
                goal = np.array([1.3, .5], dtype=np.float32) * scale
                obs, _ = env.reset(options={'init_pose': [.35*scale, .36*scale, .3],
                                             'goal': goal})
                expected = goal_vectors(obs, np.array([0]), goal[None], cfg)
                np.testing.assert_allclose(expected, obs[:, 4:6], atol=2e-7)
                corrupt = obs.copy()
                corrupt[:, 4:6] = 0
                with self.assertRaisesRegex(ValueError, 'Stale goal'):
                    require_correct_goals(corrupt, np.array([0]), goal[None], cfg)
                result = audit_goals(corrupt, np.array([0]), goal[None], cfg, repair=True)
                self.assertEqual(result['incorrect_goal_rows'], 1)
                np.testing.assert_allclose(corrupt, obs, atol=2e-7)
                env.close()

    def test_holdout_excludes_every_repeat_of_a_test_start(self):
        from scripts.il.train_goal_bc import split_by_start_pose
        poses = np.array([[.1, .2, 0], [.1, .2, 0], [.2, .3, 0],
                          [.3, .3, 0], [.3, .3, 0]])
        keep, ids = split_by_start_pose(poses, 2, 3)
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(np.unique(poses[ids], axis=0)), 2)
        self.assertFalse(any(np.array_equal(p, q)
                             for p in poses[keep] for q in poses[ids]))

    def test_invalid_reset_pose_and_goal_are_rejected(self):
        for options in ({'init_pose': [.758, .2, 0]}, {'goal': [float('nan'), .3]},
                        {'init_pose': [1, 2]}):
            with self.assertRaises(ValueError):
                self.env.reset(options=options)


if __name__ == '__main__':
    unittest.main()
