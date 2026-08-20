# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared infrastructure for LIMX tetrahedral elastic constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp
from warp.fem.linalg import symmetric_eigenvalues_qr

from ..block_csr import BlockCsrBuilder, BlockCsrMatrix


class vec9(wp.types.vector(length=9, dtype=wp.float32)):
    """Nine-vector using column-major deformation-gradient coordinates."""


class mat99(wp.types.matrix(shape=(9, 9), dtype=wp.float32)):
    """Nine-by-nine deformation-gradient Hessian."""


@wp.func
def _deformation_gradient(
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
    position_3: wp.vec3,
    inverse_rest: wp.mat33,
) -> wp.mat33:
    current_edges = wp.matrix_from_cols(
        position_1 - position_0,
        position_2 - position_0,
        position_3 - position_0,
    )
    return current_edges * inverse_rest


@wp.func
def _material_gradient(inverse_rest: wp.mat33, local_vertex: int) -> wp.vec3:
    if local_vertex == 1:
        return wp.vec3(inverse_rest[0, 0], inverse_rest[0, 1], inverse_rest[0, 2])
    if local_vertex == 2:
        return wp.vec3(inverse_rest[1, 0], inverse_rest[1, 1], inverse_rest[1, 2])
    if local_vertex == 3:
        return wp.vec3(inverse_rest[2, 0], inverse_rest[2, 1], inverse_rest[2, 2])
    return -wp.vec3(
        inverse_rest[0, 0] + inverse_rest[1, 0] + inverse_rest[2, 0],
        inverse_rest[0, 1] + inverse_rest[1, 1] + inverse_rest[2, 1],
        inverse_rest[0, 2] + inverse_rest[1, 2] + inverse_rest[2, 2],
    )


@wp.func
def _project_psd(hessian: mat99) -> mat99:
    eigenvalues, eigenvectors_by_row = symmetric_eigenvalues_qr(hessian, 1.0e-6)
    projected = mat99(0.0)
    for mode in range(9):
        eigenvalue = wp.max(eigenvalues[mode], 0.0)
        for row in range(9):
            for column in range(9):
                projected[row, column] += (
                    eigenvalue * eigenvectors_by_row[mode, row] * eigenvectors_by_row[mode, column]
                )
    return projected


@wp.func
def _map_hessian_block(
    hessian: mat99,
    material_gradient_i: wp.vec3,
    material_gradient_j: wp.vec3,
) -> wp.mat33:
    block = wp.mat33(0.0)
    for spatial_i in range(3):
        for spatial_j in range(3):
            value = float(0.0)
            for material_i in range(3):
                for material_j in range(3):
                    value += (
                        material_gradient_i[material_i]
                        * hessian[3 * material_i + spatial_i, 3 * material_j + spatial_j]
                        * material_gradient_j[material_j]
                    )
            block[spatial_i, spatial_j] = value
    return block


@wp.kernel
def _compute_minimum_determinant(
    tetrahedron_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat33],
    positions: wp.array[wp.vec3],
    minimum: wp.array[float],
):
    tetrahedron = wp.tid()
    deformation = _deformation_gradient(
        positions[tetrahedron_indices[tetrahedron, 0]],
        positions[tetrahedron_indices[tetrahedron, 1]],
        positions[tetrahedron_indices[tetrahedron, 2]],
        positions[tetrahedron_indices[tetrahedron, 3]],
        inverse_rest_matrices[tetrahedron],
    )
    wp.atomic_min(minimum, 0, wp.determinant(deformation))


class _TetrahedronElasticConstraintBase:
    """Store validated tetrahedron material data and fixed CSR bindings."""

    def __init__(
        self,
        tetrahedron_indices: Sequence[tuple[int, int, int, int]],
        inverse_rest_matrices: Sequence[wp.mat33],
        shear_moduli: Sequence[float],
        lame_parameters: Sequence[float],
        particle_count: int,
        device: Any,
    ):
        """Create a tetrahedral elastic constraint batch.

        Args:
            tetrahedron_indices: Four particle indices per tetrahedron.
            inverse_rest_matrices: Inverse rest matrices per tetrahedron [1/m].
            shear_moduli: Shear modulus per tetrahedron [Pa].
            lame_parameters: First Lamé parameter per tetrahedron [Pa].
            particle_count: Number of particles in the associated model.
            device: Warp device storing runtime arrays.
        """
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        tetrahedron_count = len(tetrahedron_indices)
        if (
            tetrahedron_count == 0
            or tetrahedron_count != len(inverse_rest_matrices)
            or tetrahedron_count != len(shear_moduli)
            or tetrahedron_count != len(lame_parameters)
        ):
            raise ValueError("Tetrahedron and material arrays must have equal nonzero length")

        self.host_tetrahedron_indices = tuple(
            tuple(int(index) for index in tetrahedron) for tetrahedron in tetrahedron_indices
        )
        self.host_inverse_rest_matrices = tuple(
            np.asarray(matrix, dtype=np.float32).reshape(3, 3) for matrix in inverse_rest_matrices
        )
        self.host_shear_moduli = tuple(float(value) for value in shear_moduli)
        self.host_lame_parameters = tuple(float(value) for value in lame_parameters)

        for tetrahedron in self.host_tetrahedron_indices:
            if len(tetrahedron) != 4 or len(set(tetrahedron)) != 4:
                raise ValueError("Tetrahedra must contain exactly four distinct particle indices")
            if any(index < 0 or index >= particle_count for index in tetrahedron):
                raise ValueError(f"Tetrahedron {tetrahedron} is outside particle_count={particle_count}")

        rest_volumes = []
        for inverse_rest in self.host_inverse_rest_matrices:
            if not np.isfinite(inverse_rest).all():
                raise ValueError("Inverse rest matrices must be finite and nonsingular")
            determinant = float(np.linalg.det(inverse_rest))
            if not np.isfinite(determinant) or abs(determinant) <= 1.0e-12:
                raise ValueError("Inverse rest matrices must be finite and nonsingular")
            rest_volume = 1.0 / (6.0 * determinant)
            if not np.isfinite(rest_volume) or rest_volume <= 0.0:
                raise ValueError("Inverse rest matrices must define positive rest volumes")
            rest_volumes.append(rest_volume)

        if any(not np.isfinite(value) or value <= 0.0 for value in self.host_shear_moduli):
            raise ValueError("Tetrahedron shear moduli must be finite and positive")
        if any(not np.isfinite(value) or value < 0.0 for value in self.host_lame_parameters):
            raise ValueError("Tetrahedron Lamé parameters must be finite and nonnegative")

        self.host_rest_volumes = tuple(rest_volumes)
        self.particle_count = particle_count
        self.device = wp.get_device(device)
        self.tetrahedron_indices = wp.array2d(self.host_tetrahedron_indices, dtype=int, device=self.device)
        self.inverse_rest_matrices = wp.array(
            np.asarray(self.host_inverse_rest_matrices), dtype=wp.mat33, device=self.device
        )
        self.rest_volumes = wp.array(self.host_rest_volumes, dtype=float, device=self.device)
        self.shear_moduli = wp.array(self.host_shear_moduli, dtype=float, device=self.device)
        self.lame_parameters = wp.array(self.host_lame_parameters, dtype=float, device=self.device)
        self.hessian_block_indices: wp.array2d[int] | None = None
        self.hessian_value_count: int | None = None
        self._minimum_determinant = wp.empty(1, dtype=float, device=self.device)

    def append_hessian_structure(self, builder: BlockCsrBuilder) -> None:
        """Append all sixteen ordered particle-pair blocks per tetrahedron."""
        if builder.row_count != self.particle_count:
            raise ValueError("Constraint and block matrix particle counts differ")
        builder.ensure_stencil_blocks(np.asarray(self.host_tetrahedron_indices, dtype=np.int32))

    def bind_hessian(self, matrix: BlockCsrMatrix) -> None:
        """Bind tetrahedron blocks to finalized block-CSR value indices."""
        if matrix.row_count != self.particle_count or matrix.device != self.device:
            raise ValueError("Constraint and block matrix must have matching particle counts and devices")
        block_indices = matrix.stencil_block_indices(np.asarray(self.host_tetrahedron_indices, dtype=np.int32))
        self.hessian_block_indices = wp.array2d(block_indices, dtype=int, device=self.device)
        self.hessian_value_count = len(matrix.values)

    def minimum_determinant(self, positions: wp.array[wp.vec3]) -> float:
        """Return the smallest current deformation determinant."""
        self._validate_positions(positions)
        self._minimum_determinant.fill_(np.finfo(np.float32).max)
        wp.launch(
            _compute_minimum_determinant,
            dim=len(self.rest_volumes),
            inputs=[self.tetrahedron_indices, self.inverse_rest_matrices, positions],
            outputs=[self._minimum_determinant],
            device=self.device,
        )
        return float(self._minimum_determinant.numpy()[0])

    def _validate_positions(self, positions: wp.array[wp.vec3]) -> None:
        if positions.dtype != wp.vec3 or len(positions) != self.particle_count:
            raise ValueError(f"Positions must contain {self.particle_count} wp.vec3 rows")
        if positions.device != self.device:
            raise ValueError("Constraint and positions must use the same device")

    def _validate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        self._validate_positions(positions)
        if output.dtype != wp.vec3 or len(output) != self.particle_count or output.device != self.device:
            raise ValueError(f"Forces must contain {self.particle_count} wp.vec3 rows on the constraint device")

    def _validate_hessian(self, hessian_values: wp.array[wp.mat33]) -> None:
        if self.hessian_block_indices is None:
            raise RuntimeError("bind_hessian() must be called before Hessian assembly")
        if (
            hessian_values.dtype != wp.mat33
            or hessian_values.device != self.device
            or len(hessian_values) != self.hessian_value_count
        ):
            raise ValueError(f"Expected {self.hessian_value_count} wp.mat33 Hessian blocks")

    def _validate_energy(
        self,
        positions: wp.array[wp.vec3],
        output: wp.array[float],
        invalid_count: wp.array[int],
    ) -> None:
        self._validate_positions(positions)
        if output.dtype != wp.float32 or len(output) != 1 or output.device != self.device:
            raise ValueError("Energy output must be one float on the constraint device")
        if invalid_count.dtype != wp.int32 or len(invalid_count) != 1 or invalid_count.device != self.device:
            raise ValueError("Invalid count must be one integer on the constraint device")
