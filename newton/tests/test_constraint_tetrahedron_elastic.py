# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for LIMX tetrahedral elastic constraints."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.tetrahedron_linear_elastic import (
    ConstraintTetrahedronLinearElastic,
)

REST_POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
TETRAHEDRA = [(0, 1, 2, 3)]
INVERSE_REST_MATRICES = [wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)]
DEFORMED_POSITIONS = np.asarray(
    [
        [0.10, -0.05, 0.02],
        [1.18, 0.08, -0.03],
        [0.04, 0.91, 0.12],
        [-0.06, 0.10, 1.09],
    ],
    dtype=np.float32,
)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintTetrahedronLinearElastic(unittest.TestCase):
    @staticmethod
    def make_constraint() -> ConstraintTetrahedronLinearElastic:
        return ConstraintTetrahedronLinearElastic(
            TETRAHEDRA,
            INVERSE_REST_MATRICES,
            [4.0],
            [7.0],
            4,
            "cuda:0",
        )

    @classmethod
    def evaluate_energy(cls, positions: np.ndarray) -> float:
        constraint = cls.make_constraint()
        positions_wp = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        energy = wp.zeros(1, dtype=float, device="cuda:0")
        invalid_count = wp.zeros(1, dtype=int, device="cuda:0")
        constraint.accumulate_energy(positions_wp, energy, invalid_count)
        if int(invalid_count.numpy()[0]) != 0:
            raise AssertionError("Quadratic energy unexpectedly marked a valid state invalid")
        return float(energy.numpy()[0])

    @classmethod
    def assemble(cls, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        constraint = cls.make_constraint()
        builder = BlockCsrBuilder(4)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize("cuda:0")
        constraint.bind_hessian(matrix)
        positions_wp = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
        matrix.clear_values()
        constraint.accumulate_force_and_hessian(positions_wp, force, matrix.values)
        values = matrix.values.numpy()
        dense = np.empty((12, 12), dtype=np.float32)
        for row in range(4):
            for column in range(4):
                dense[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] = values[matrix.block_index(row, column)]
        return force.numpy(), dense

    def test_rest_state_has_zero_energy_and_force(self):
        """Keep quadratic tetrahedra force-free at the rest state."""
        with wp.ScopedDevice("cuda:0"):
            constraint = ConstraintTetrahedronLinearElastic(
                TETRAHEDRA,
                INVERSE_REST_MATRICES,
                [4.0],
                [7.0],
                4,
                "cuda:0",
            )
            positions = wp.array(REST_POSITIONS, dtype=wp.vec3, device="cuda:0")
            force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
            energy = wp.zeros(1, dtype=float, device="cuda:0")
            invalid_count = wp.zeros(1, dtype=int, device="cuda:0")

            constraint.accumulate_force(positions, force)
            constraint.accumulate_energy(positions, energy, invalid_count)

            np.testing.assert_allclose(force.numpy(), 0.0, atol=1.0e-6)
            self.assertAlmostEqual(float(energy.numpy()[0]), 0.0, places=6)
            self.assertEqual(int(invalid_count.numpy()[0]), 0)

    def test_rejects_invalid_material_parameters(self):
        """Reject nonpositive shear modulus and negative Lamé parameter."""
        with wp.ScopedDevice("cuda:0"):
            with self.assertRaisesRegex(ValueError, "shear moduli"):
                ConstraintTetrahedronLinearElastic(
                    TETRAHEDRA,
                    INVERSE_REST_MATRICES,
                    [0.0],
                    [7.0],
                    4,
                    "cuda:0",
                )
            with self.assertRaisesRegex(ValueError, "Lamé"):
                ConstraintTetrahedronLinearElastic(
                    TETRAHEDRA,
                    INVERSE_REST_MATRICES,
                    [4.0],
                    [-1.0],
                    4,
                    "cuda:0",
                )

    def test_force_matches_negative_energy_gradient(self):
        """Match quadratic tetrahedral force to centered energy differences."""
        with wp.ScopedDevice("cuda:0"):
            analytical_force, _ = self.assemble(DEFORMED_POSITIONS)
            numerical_force = np.empty((4, 3), dtype=np.float32)
            step = 1.0e-3
            for vertex in range(4):
                for axis in range(3):
                    positive = DEFORMED_POSITIONS.copy()
                    negative = DEFORMED_POSITIONS.copy()
                    positive[vertex, axis] += step
                    negative[vertex, axis] -= step
                    numerical_force[vertex, axis] = -(
                        self.evaluate_energy(positive) - self.evaluate_energy(negative)
                    ) / (2.0 * step)

        np.testing.assert_allclose(analytical_force, numerical_force, rtol=2.0e-3, atol=2.0e-3)

    def test_assembled_hessian_is_constant_symmetric_psd(self):
        """Assemble a constant symmetric PSD quadratic tetrahedral Hessian."""
        with wp.ScopedDevice("cuda:0"):
            _, rest_hessian = self.assemble(REST_POSITIONS)
            _, deformed_hessian = self.assemble(DEFORMED_POSITIONS)

        np.testing.assert_allclose(rest_hessian, deformed_hessian, atol=1.0e-6)
        np.testing.assert_allclose(rest_hessian, rest_hessian.T, atol=1.0e-6)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(rest_hessian).min()), -1.0e-5)
        translation = np.tile([0.3, -0.2, 0.7], 4)
        np.testing.assert_allclose(rest_hessian @ translation, 0.0, atol=1.0e-5)

    def test_reports_current_deformation_determinant(self):
        """Report the current tetrahedron deformation determinant."""
        with wp.ScopedDevice("cuda:0"):
            constraint = self.make_constraint()
            positions = wp.array(DEFORMED_POSITIONS, dtype=wp.vec3, device="cuda:0")
            result = constraint.minimum_determinant(positions)
        edges = np.column_stack([DEFORMED_POSITIONS[index] - DEFORMED_POSITIONS[0] for index in (1, 2, 3)])
        self.assertAlmostEqual(result, float(np.linalg.det(edges)), places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
