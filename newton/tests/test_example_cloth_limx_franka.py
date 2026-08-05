# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
