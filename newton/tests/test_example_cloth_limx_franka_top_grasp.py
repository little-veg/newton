# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.viewer import ViewerNull


class TestClothLimxFrankaTopGrasp(unittest.TestCase):
    def test_builds_centered_top_down_grasp_sequence(self):
        """Build a centered top-down descend, pinch, lift, hold, and release sequence."""
        module_name = "newton.examples.cloth.example_cloth_limx_franka_top_grasp"
        self.assertIsNotNone(importlib.util.find_spec(module_name))
        module = importlib.import_module(module_name)
        poses = module._create_top_grasp_poses()

        self.assertEqual(poses.shape, (8, 9))
        np.testing.assert_allclose(poses[:, 1:3], [(0.0, -0.5)] * 8, atol=0.0)
        np.testing.assert_allclose(poses[:, 4:8], [(1.0, 0.0, 0.0, 0.0)] * 8, atol=0.0)
        np.testing.assert_allclose(poses[:, 0], (0.5, 0.8, 0.8, 0.4, 1.2, 0.8, 0.8, 0.6), atol=0.0)
        self.assertAlmostEqual(float(poses[0, 3]), 0.35)
        self.assertAlmostEqual(float(poses[2, 3]), 0.211)
        self.assertAlmostEqual(float(poses[4, 3]), 0.42)
        self.assertAlmostEqual(float(poses[0, 8]), 0.04)
        self.assertAlmostEqual(float(poses[2, 8]), 0.0029)
        self.assertAlmostEqual(float(poses[6, 8]), 0.04)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxFrankaTopGraspCuda(unittest.TestCase):
    def test_reaches_grasp_pose_without_entering_table(self):
        """Reach the centered top grasp with both complete finger boxes above the table."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka_top_grasp")
        self.assertTrue(hasattr(module, "Example"))
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=True))
            while example.sim_time <= example.descend_end_time + example.frame_dt:
                example.step()
            finger_positions = example.gripper_contact.collider_positions.numpy()
            tcp_position = example.tcp_position.numpy()[0]

        np.testing.assert_allclose(tcp_position, example.grasp_position, atol=0.01)
        self.assertGreater(float(finger_positions[:, 2].min()), module.TABLE_TOP_Z + 0.004)
        self.assertLess(float(finger_positions[:, 2].min()), module.CLOTH_HEIGHT + 0.002)
        np.testing.assert_allclose(example.cloth_rest_positions[:, :2].mean(axis=0), (0.0, -0.5), atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
