# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for LIMX tetrahedral elastic constraints."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.tetrahedron_elastic_common import _project_psd, mat99
from newton._src.solvers.limx.constraints.tetrahedron_linear_elastic import (
    ConstraintTetrahedronLinearElastic,
)
from newton._src.solvers.limx.constraints.tetrahedron_neo_hookean import (
    ConstraintTetrahedronNeoHookean,
    _neo_hookean_energy,
    _neo_hookean_gradient,
    _neo_hookean_hessian,
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


@wp.kernel
def _evaluate_neo_hookean_material(
    deformation: wp.mat33,
    shear_modulus: float,
    lame_parameter: float,
    rest_volume: float,
    energy: wp.array[float],
    gradient: wp.array[wp.mat33],
    hessian: wp.array[mat99],
    projected_hessian: wp.array[mat99],
):
    energy[0] = _neo_hookean_energy(deformation, shear_modulus, lame_parameter, rest_volume)
    gradient[0] = _neo_hookean_gradient(deformation, shear_modulus, lame_parameter, rest_volume)
    exact_hessian = _neo_hookean_hessian(deformation, shear_modulus, lame_parameter, rest_volume)
    hessian[0] = exact_hessian
    projected_hessian[0] = _project_psd(exact_hessian)


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


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintTetrahedronNeoHookean(unittest.TestCase):
    @staticmethod
    def make_constraint() -> ConstraintTetrahedronNeoHookean:
        return ConstraintTetrahedronNeoHookean(
            TETRAHEDRA,
            INVERSE_REST_MATRICES,
            [4.0],
            [7.0],
            4,
            "cuda:0",
        )

    @classmethod
    def evaluate_energy(cls, positions: np.ndarray) -> tuple[float, int]:
        constraint = cls.make_constraint()
        positions_wp = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        energy = wp.zeros(1, dtype=float, device="cuda:0")
        invalid_count = wp.zeros(1, dtype=int, device="cuda:0")
        constraint.accumulate_energy(positions_wp, energy, invalid_count)
        return float(energy.numpy()[0]), int(invalid_count.numpy()[0])

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

    @staticmethod
    def evaluate_material(deformation: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        energy = wp.empty(1, dtype=float, device="cuda:0")
        gradient = wp.empty(1, dtype=wp.mat33, device="cuda:0")
        hessian = wp.empty(1, dtype=mat99, device="cuda:0")
        projected_hessian = wp.empty(1, dtype=mat99, device="cuda:0")
        wp.launch(
            _evaluate_neo_hookean_material,
            dim=1,
            inputs=[wp.mat33(*deformation.reshape(-1)), 4.0, 7.0, 1.0 / 6.0],
            outputs=[energy, gradient, hessian, projected_hessian],
            device="cuda:0",
        )
        return (
            float(energy.numpy()[0]),
            gradient.numpy()[0],
            hessian.numpy()[0],
            projected_hessian.numpy()[0],
        )

    def test_rest_state_has_zero_energy_and_force(self):
        """Keep logarithmic Neo-Hookean tetrahedra stress-free at rest."""
        with wp.ScopedDevice("cuda:0"):
            constraint = self.make_constraint()
            positions = wp.array(REST_POSITIONS, dtype=wp.vec3, device="cuda:0")
            force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
            constraint.accumulate_force(positions, force)
            energy, invalid_count = self.evaluate_energy(REST_POSITIONS)

        self.assertAlmostEqual(energy, 0.0, places=6)
        self.assertEqual(invalid_count, 0)
        np.testing.assert_allclose(force.numpy(), 0.0, atol=1.0e-6)

    def test_force_matches_negative_energy_gradient(self):
        """Match logarithmic Neo-Hookean force to centered energy differences."""
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
                    positive_energy, positive_invalid = self.evaluate_energy(positive)
                    negative_energy, negative_invalid = self.evaluate_energy(negative)
                    self.assertEqual((positive_invalid, negative_invalid), (0, 0))
                    numerical_force[vertex, axis] = -(positive_energy - negative_energy) / (2.0 * step)

        np.testing.assert_allclose(analytical_force, numerical_force, rtol=3.0e-3, atol=3.0e-3)

    def test_exact_material_hessian_matches_gradient_difference(self):
        """Match the complete unprojected material Hessian to gradient differences."""
        deformation = np.asarray(
            [[1.12, 0.08, -0.03], [0.05, 0.91, 0.07], [-0.02, 0.11, 1.04]],
            dtype=np.float32,
        )
        with wp.ScopedDevice("cuda:0"):
            _, _, analytical_hessian, _ = self.evaluate_material(deformation)
            numerical_hessian = np.empty((9, 9), dtype=np.float32)
            step = 1.0e-3
            for material in range(3):
                for spatial in range(3):
                    positive = deformation.copy()
                    negative = deformation.copy()
                    positive[spatial, material] += step
                    negative[spatial, material] -= step
                    _, positive_gradient, _, _ = self.evaluate_material(positive)
                    _, negative_gradient, _, _ = self.evaluate_material(negative)
                    numerical_hessian[:, 3 * material + spatial] = (
                        positive_gradient.reshape(-1, order="F") - negative_gradient.reshape(-1, order="F")
                    ) / (2.0 * step)

        np.testing.assert_allclose(analytical_hessian, numerical_hessian, rtol=3.0e-3, atol=3.0e-3)

    def test_projected_hessian_matches_numpy_eigendecomposition(self):
        """Match full material-space eigenvalue clamping and reconstruction."""
        deformation = np.asarray(
            [[0.35, 0.11, 0.02], [0.05, 1.07, -0.04], [0.01, 0.08, 0.92]],
            dtype=np.float32,
        )
        with wp.ScopedDevice("cuda:0"):
            _, _, exact_hessian, projected_hessian = self.evaluate_material(deformation)
        eigenvalues, eigenvectors = np.linalg.eigh(exact_hessian)
        expected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        np.testing.assert_allclose(projected_hessian, expected, rtol=3.0e-3, atol=3.0e-3)

    def test_assembled_hessian_is_symmetric_psd_with_translation_nullspace(self):
        """Preserve symmetry, PSD behavior, and translations after particle mapping."""
        with wp.ScopedDevice("cuda:0"):
            force, hessian = self.assemble(DEFORMED_POSITIONS)
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-5)
        np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-5)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(hessian).min()), -2.0e-4)
        translation = np.tile([0.3, -0.2, 0.7], 4)
        np.testing.assert_allclose(hessian @ translation, 0.0, atol=2.0e-4)

    def test_inverted_state_is_invalid_without_nan_force(self):
        """Reject nonpositive determinants before logarithm or inverse evaluation."""
        inverted_positions = REST_POSITIONS.copy()
        inverted_positions[[1, 2]] = inverted_positions[[2, 1]]
        with wp.ScopedDevice("cuda:0"):
            energy, invalid_count = self.evaluate_energy(inverted_positions)
            constraint = self.make_constraint()
            positions = wp.array(inverted_positions, dtype=wp.vec3, device="cuda:0")
            force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
            constraint.accumulate_force(positions, force)
        self.assertEqual(invalid_count, 1)
        self.assertEqual(energy, 0.0)
        self.assertTrue(np.isfinite(force.numpy()).all())
        np.testing.assert_array_equal(force.numpy(), np.zeros((4, 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
