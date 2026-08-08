# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import math
import unittest

import numpy as np
import warp as wp

from newton.viewer import ViewerNull


class TestClothLimxFold(unittest.TestCase):
    def test_builds_100_cell_grid_and_half_circle_fold_targets(self):
        """Build the requested grid and drive its right edge through a half fold."""
        module_name = "newton.examples.cloth.example_cloth_limx_fold"
        self.assertIsNotNone(importlib.util.find_spec(module_name))
        module = importlib.import_module(module_name)

        positions, triangles = module._create_square_cloth_grid(100, 1.0, 0.205)
        right_rest = positions[100::101]
        start = module._compute_fold_boundary_targets(right_rest, 0.0, 0.0)
        midpoint = module._compute_fold_boundary_targets(right_rest, 0.5 * math.pi, 0.5)
        finish = module._compute_fold_boundary_targets(right_rest, math.pi, 1.0)

        self.assertEqual(positions.shape, (10201, 3))
        self.assertEqual(triangles.shape, (20000, 3))
        np.testing.assert_allclose(start[:, 0], 0.5, atol=1.0e-7)
        np.testing.assert_allclose(start[:, 2], 0.205, atol=1.0e-7)
        np.testing.assert_allclose(midpoint[:, 0], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(midpoint[:, 2], 0.708, atol=1.0e-6)
        np.testing.assert_allclose(finish[:, 0], -0.5, atol=1.0e-6)
        np.testing.assert_allclose(finish[:, 2], 0.211, atol=1.0e-6)
        np.testing.assert_allclose(finish[:, 1], right_rest[:, 1], atol=1.0e-7)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxFoldCuda(unittest.TestCase):
    def test_scene_starts_supported_and_advances_one_step(self):
        """Start the full-resolution cloth on its table and advance finite state."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_fold")
        self.assertTrue(hasattr(module, "Example"))
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), None)
            example.step()
            positions = example.state_0.particle_q.numpy()
            velocities = example.state_0.particle_qd.numpy()

        self.assertEqual(example.model.particle_count, 10201)
        self.assertEqual(len(example.left_boundary_indices), 101)
        self.assertEqual(len(example.right_boundary_indices), 101)
        self.assertAlmostEqual(example.sim_dt, 0.01)
        self.assertAlmostEqual(example.table_contact.friction, 0.05)
        self.assertFalse(example.table_contact.enable_ccd)
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(velocities).all())
        self.assertGreater(float(positions[:, 2].min()), module.TABLE_TOP_Z)


if __name__ == "__main__":
    unittest.main()
