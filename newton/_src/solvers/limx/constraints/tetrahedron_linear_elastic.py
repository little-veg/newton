# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quadratic small-strain tetrahedral elasticity for LIMX."""

from __future__ import annotations

import warp as wp

from .tetrahedron_elastic_common import (
    _deformation_gradient,
    _map_hessian_block,
    _material_gradient,
    _TetrahedronElasticConstraintBase,
    mat99,
)


@wp.func
def _linear_elastic_strain(deformation: wp.mat33) -> wp.mat33:
    return 0.5 * (deformation + wp.transpose(deformation)) - wp.identity(3, float)


@wp.func
def _linear_elastic_energy(
    deformation: wp.mat33,
    shear_modulus: float,
    lame_parameter: float,
    rest_volume: float,
) -> float:
    strain = _linear_elastic_strain(deformation)
    trace = wp.trace(strain)
    return rest_volume * (shear_modulus * wp.ddot(strain, strain) + 0.5 * lame_parameter * trace * trace)


@wp.func
def _linear_elastic_gradient(
    deformation: wp.mat33,
    shear_modulus: float,
    lame_parameter: float,
    rest_volume: float,
) -> wp.mat33:
    strain = _linear_elastic_strain(deformation)
    return rest_volume * (2.0 * shear_modulus * strain + lame_parameter * wp.trace(strain) * wp.identity(3, float))


@wp.func
def _linear_elastic_hessian(
    shear_modulus: float,
    lame_parameter: float,
    rest_volume: float,
) -> mat99:
    hessian = mat99(0.0)
    for material_i in range(3):
        for spatial_i in range(3):
            row = 3 * material_i + spatial_i
            for material_j in range(3):
                for spatial_j in range(3):
                    column = 3 * material_j + spatial_j
                    value = float(0.0)
                    if spatial_i == spatial_j and material_i == material_j:
                        value += shear_modulus
                    if material_i == spatial_j and spatial_i == material_j:
                        value += shear_modulus
                    if spatial_i == material_i and spatial_j == material_j:
                        value += lame_parameter
                    hessian[row, column] = rest_volume * value
    return hessian


@wp.kernel
def _accumulate_linear_elastic_energy(
    tetrahedron_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat33],
    rest_volumes: wp.array[float],
    shear_moduli: wp.array[float],
    lame_parameters: wp.array[float],
    positions: wp.array[wp.vec3],
    output: wp.array[float],
):
    tetrahedron = wp.tid()
    deformation = _deformation_gradient(
        positions[tetrahedron_indices[tetrahedron, 0]],
        positions[tetrahedron_indices[tetrahedron, 1]],
        positions[tetrahedron_indices[tetrahedron, 2]],
        positions[tetrahedron_indices[tetrahedron, 3]],
        inverse_rest_matrices[tetrahedron],
    )
    energy = _linear_elastic_energy(
        deformation,
        shear_moduli[tetrahedron],
        lame_parameters[tetrahedron],
        rest_volumes[tetrahedron],
    )
    wp.atomic_add(output, 0, energy)


@wp.kernel
def _accumulate_linear_elastic_force(
    tetrahedron_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat33],
    rest_volumes: wp.array[float],
    shear_moduli: wp.array[float],
    lame_parameters: wp.array[float],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
):
    tetrahedron = wp.tid()
    inverse_rest = inverse_rest_matrices[tetrahedron]
    deformation = _deformation_gradient(
        positions[tetrahedron_indices[tetrahedron, 0]],
        positions[tetrahedron_indices[tetrahedron, 1]],
        positions[tetrahedron_indices[tetrahedron, 2]],
        positions[tetrahedron_indices[tetrahedron, 3]],
        inverse_rest,
    )
    gradient = _linear_elastic_gradient(
        deformation,
        shear_moduli[tetrahedron],
        lame_parameters[tetrahedron],
        rest_volumes[tetrahedron],
    )
    for local_vertex in range(4):
        particle = tetrahedron_indices[tetrahedron, local_vertex]
        wp.atomic_sub(forces, particle, gradient * _material_gradient(inverse_rest, local_vertex))


@wp.kernel
def _accumulate_linear_elastic_force_and_hessian(
    tetrahedron_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat33],
    rest_volumes: wp.array[float],
    shear_moduli: wp.array[float],
    lame_parameters: wp.array[float],
    hessian_block_indices: wp.array2d[int],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_values: wp.array[wp.mat33],
):
    tetrahedron = wp.tid()
    inverse_rest = inverse_rest_matrices[tetrahedron]
    deformation = _deformation_gradient(
        positions[tetrahedron_indices[tetrahedron, 0]],
        positions[tetrahedron_indices[tetrahedron, 1]],
        positions[tetrahedron_indices[tetrahedron, 2]],
        positions[tetrahedron_indices[tetrahedron, 3]],
        inverse_rest,
    )
    gradient = _linear_elastic_gradient(
        deformation,
        shear_moduli[tetrahedron],
        lame_parameters[tetrahedron],
        rest_volumes[tetrahedron],
    )
    hessian = _linear_elastic_hessian(
        shear_moduli[tetrahedron],
        lame_parameters[tetrahedron],
        rest_volumes[tetrahedron],
    )
    for local_i in range(4):
        material_gradient_i = _material_gradient(inverse_rest, local_i)
        wp.atomic_sub(
            forces,
            tetrahedron_indices[tetrahedron, local_i],
            gradient * material_gradient_i,
        )
        for local_j in range(4):
            block = _map_hessian_block(
                hessian,
                material_gradient_i,
                _material_gradient(inverse_rest, local_j),
            )
            block_index = hessian_block_indices[tetrahedron, 4 * local_i + local_j]
            wp.atomic_add(hessian_values, block_index, block)


class ConstraintTetrahedronLinearElastic(_TetrahedronElasticConstraintBase):
    """A batch of quadratic small-strain tetrahedral constraints."""

    def accumulate_energy(
        self,
        positions: wp.array[wp.vec3],
        output: wp.array[float],
        invalid_count: wp.array[int],
    ) -> None:
        """Add quadratic elastic energy evaluated at ``positions``."""
        self._validate_energy(positions, output, invalid_count)
        wp.launch(
            _accumulate_linear_elastic_energy,
            dim=len(self.rest_volumes),
            inputs=[
                self.tetrahedron_indices,
                self.inverse_rest_matrices,
                self.rest_volumes,
                self.shear_moduli,
                self.lame_parameters,
                positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add quadratic elastic forces evaluated at ``positions``."""
        self._validate_force(positions, output)
        wp.launch(
            _accumulate_linear_elastic_force,
            dim=len(self.rest_volumes),
            inputs=[
                self.tetrahedron_indices,
                self.inverse_rest_matrices,
                self.rest_volumes,
                self.shear_moduli,
                self.lame_parameters,
                positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_force_and_hessian(
        self,
        positions: wp.array[wp.vec3],
        force_output: wp.array[wp.vec3],
        hessian_values: wp.array[wp.mat33],
    ) -> None:
        """Add quadratic forces and exact constant Hessian blocks."""
        self._validate_force(positions, force_output)
        self._validate_hessian(hessian_values)
        wp.launch(
            _accumulate_linear_elastic_force_and_hessian,
            dim=len(self.rest_volumes),
            inputs=[
                self.tetrahedron_indices,
                self.inverse_rest_matrices,
                self.rest_volumes,
                self.shear_moduli,
                self.lame_parameters,
                self.hessian_block_indices,
                positions,
            ],
            outputs=[force_output, hessian_values],
            device=self.device,
        )
