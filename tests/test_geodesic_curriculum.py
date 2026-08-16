from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ant_swarm import (AntSwarmEnv, PoseGeodesicField, curriculum_anchors,
                       load_config)


def write_field(path: Path, dist: np.ndarray) -> None:
    reachable = np.ones_like(dist, dtype=bool)
    np.savez_compressed(
        path,
        dist=np.asarray(dist, dtype=np.uint16),
        reachable=reachable,
        norm=np.float64(max(int(np.max(dist)), 1)),
        x0=np.float64(0.0), dx=np.float64(1.0),
        y0=np.float64(0.0), dy=np.float64(1.0),
        th0=np.float64(-180.0), dth=np.float64(90.0),
    )


class GeodesicPathTest(unittest.TestCase):
    def test_path_descends_one_bfs_step_and_wraps_theta(self):
        with tempfile.TemporaryDirectory() as directory:
            field_path = Path(directory) / "theta.npz"
            # The last theta bin reaches bin zero through the periodic boundary.
            write_field(field_path, np.asarray([0, 1, 2, 1])[:, None, None])
            field = PoseGeodesicField(field_path)
            start = field.pose((3, 0, 0))
            path = field.path_from(start)
            steps = [int(field.steps[field.index(*map(float, p))]) for p in path]
            self.assertEqual(steps, [1, 0])
            self.assertTrue(np.all(path[:, 2] >= -math.pi))
            self.assertTrue(np.all(path[:, 2] < math.pi))

    def test_curriculum_anchors_are_easy_to_hard_on_one_path(self):
        with tempfile.TemporaryDirectory() as directory:
            field_path = Path(directory) / "line.npz"
            line = np.arange(11, dtype=np.uint16)[::-1]
            write_field(field_path, np.broadcast_to(line, (4, 1, len(line))))
            field = PoseGeodesicField(field_path)
            start = field.pose((0, 0, 0))
            anchors = curriculum_anchors(field, start, stages=6,
                                         near_goal_fraction=0.2)
            steps = [int(field.steps[field.index(*map(float, p))])
                     for p in anchors]
            self.assertEqual(steps[-1], 10)
            self.assertTrue(all(a < b for a, b in zip(steps, steps[1:])))
            np.testing.assert_allclose(anchors[-1], start)

    def test_unreachable_start_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            field_path = Path(directory) / "blocked.npz"
            write_field(field_path, np.asarray([[[1, 0]]] * 4))
            with np.load(field_path) as saved:
                payload = {name: saved[name] for name in saved.files}
            payload["reachable"][:, :, 0] = False
            np.savez_compressed(field_path, **payload)
            field = PoseGeodesicField(field_path)
            with self.assertRaises(ValueError):
                field.path_from(field.pose((0, 0, 0)))


class ExactSpawnAndVelocityTest(unittest.TestCase):
    @staticmethod
    def config():
        cfg = load_config("configs/rl/pnas_dyn_geo_v2.yaml")
        # These unit tests exercise the environment without requiring a field.
        cfg.env.reward_mode = "sparse"
        return cfg

    def test_exact_spawn_is_tagged_and_resets_momentum(self):
        cfg = self.config()
        env = AntSwarmEnv(cfg, seed=7)
        pose = np.asarray([0.30, 0.36, math.pi], dtype=np.float32)
        env.set_spawn_pose(pose, stage_idx=4)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 27))
        np.testing.assert_allclose(env.state.obj.center, pose[:2])
        self.assertAlmostEqual(env.state.obj.angle, float(pose[2]))
        np.testing.assert_array_equal(env.state.obj.vel, np.zeros(2))
        self.assertEqual(env.state.obj.ang_vel, 0.0)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        self.assertEqual(info["curriculum_stage"], 4)
        np.testing.assert_allclose(info["spawn_pose"], pose, atol=1e-6)

    def test_velocity_gate_preserves_legacy_shape(self):
        cfg = self.config()
        cfg.env.observe_linear_velocity = False
        env = AntSwarmEnv(cfg, seed=0)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 25))


if __name__ == "__main__":
    unittest.main()
