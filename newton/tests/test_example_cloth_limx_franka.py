# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


class TestClothLimxFranka(unittest.TestCase):
    def test_square_cloth_grid_has_expected_flat_topology(self):
        """Build a flat square grid with alternating triangle diagonals."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        positions, triangles = module._create_square_cloth_grid(
            grid_cells=20,
            width=0.4,
            center=(0.0, -0.5),
            height=0.205,
        )

        self.assertEqual(positions.shape, (441, 3))
        self.assertEqual(triangles.shape, (800, 3))
        self.assertEqual(positions.dtype, np.float32)
        self.assertEqual(triangles.dtype, np.int32)
        np.testing.assert_allclose(positions[:, 2], 0.205, rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(positions[:, :2].min(axis=0), (-0.2, -0.7), rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(positions[:, :2].max(axis=0), (0.2, -0.3), rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(triangles[:2], ((0, 1, 22), (0, 22, 21)))


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxFrankaCuda(unittest.TestCase):
    def test_scene_keeps_square_cloth_flat_and_inactive(self):
        """Keep every square-cloth particle fixed above the table."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=False))
            flags = example.model.particle_flags.numpy()
            positions = example.state_0.particle_q.numpy()

        self.assertEqual(example.model.particle_count, 441)
        self.assertGreater(example.model.body_count, 0)
        self.assertTrue(example.model.body_label[example.hand_body].endswith("fr3_hand"))
        self.assertTrue(np.all((flags & int(newton.ParticleFlags.ACTIVE)) == 0))
        np.testing.assert_allclose(positions, example.cloth_rest_positions, rtol=0.0, atol=1.0e-7)
        self.assertGreater(float(positions[:, 2].min()), example.table_top_z)

    def test_grasp_sequence_reaches_and_lifts_from_cloth(self):
        """Drive the Franka through approach, close, and lift poses."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            frame_count = int(np.ceil(module.SEQUENCE_DURATION * module.FPS)) + 1
            example = module.Example(ViewerNull(num_frames=frame_count), SimpleNamespace(graph_capture=False))
            for _ in range(frame_count):
                example.step()
                example.test_post_step()
            example.test_final()

        self.assertLess(example.minimum_grasp_error, 0.03)
        self.assertGreater(example.maximum_tcp_height, example.grasp_position[2] + 0.10)


if __name__ == "__main__":
    unittest.main()
