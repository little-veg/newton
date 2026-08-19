# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Logarithmic compressible Neo-Hookean tetrahedra for LIMX."""

from __future__ import annotations

import warp as wp

from .tetrahedron_elastic_common import (
    _deformation_gradient,
    _map_hessian_block,
    _material_gradient,
    _project_psd,
    _TetrahedronElasticConstraintBase,
    mat99,
)


@wp.func
def _inverse_transpose(deformation: wp.mat33, determinant: float) -> wp.mat33:
    column_0 = wp.vec3(deformation[0, 0], deformation[1, 0], deformation[2, 0])
    column_1 = wp.vec3(deformation[0, 1], deformation[1, 1], deformation[2, 1])
    column_2 = wp.vec3(deformation[0, 2], deformation[1, 2], deformation[2, 2])
    return (
        wp.matrix_from_cols(
            wp.cross(column_1, column_2),
            wp.cross(column_2, column_0),
            wp.cross(column_0, column_1),
        )
        / determinant
    )


@wp.func
def _neo_hookean_energy(
    deformation: wp.mat33,
    shear_modulus: float,
    lame_parameter: float,
    rest_volume: float,
) -> float:
    log_determinant = wp.log(wp.determinant(deformation))
    return rest_volume * (
        0.5 * shear_modulus * (wp.ddot(deformation, deformation) - 3.0)
        - shear_modulus * log_determinant
        + 0.5 * lame_parameter * log_determinant * log_determinant
    )


@wp.func
def _neo_hookean_gradient(
    deformation: wp.mat33,
    shear_modulus: float,
    lame_parameter: float,
    rest_volume: float,
) -> wp.mat33:
    determinant = wp.determinant(deformation)
    inverse_transpose = _inverse_transpose(deformation, determinant)
    pressure = lame_parameter * wp.log(determinant) - shear_modulus
    return rest_volume * (shear_modulus * deformation + pressure * inverse_transpose)


@wp.func
def _neo_hookean_hessian(
    deformation: wp.mat33,
    shear_modulus: float,
    lame_parameter: float,
    rest_volume: float,
) -> mat99:
    determinant = wp.determinant(deformation)
    inverse_transpose = _inverse_transpose(deformation, determinant)
    pressure = lame_parameter * wp.log(determinant) - shear_modulus
    hessian = mat99(0.0)
    for material_i in range(3):
        for spatial_i in range(3):
            row = 3 * material_i + spatial_i
            for material_j in range(3):
                for spatial_j in range(3):
                    column = 3 * material_j + spatial_j
                    value = (
                        lame_parameter
                        * inverse_transpose[spatial_i, material_i]
                        * inverse_transpose[spatial_j, material_j]
                        - pressure * inverse_transpose[spatial_i, material_j] * inverse_transpose[spatial_j, material_i]
                    )
                    if spatial_i == spatial_j and material_i == material_j:
                        value += shear_modulus
                    hessian[row, column] = rest_volume * value
    return hessian


@wp.kernel
def _accumulate_neo_hookean_energy(
    tetrahedron_indices: wp.array2d[int],
    inverse_rest_matrices: wp.array[wp.mat33],
    rest_volumes: wp.array[float],
    shear_moduli: wp.array[float],
    lame_parameters: wp.array[float],
    positions: wp.array[wp.vec3],
    output: wp.array[float],
    invalid_count: wp.array[int],
):
    tetrahedron = wp.tid()
    deformation = _deformation_gradient(
        positions[tetrahedron_indices[tetrahedron, 0]],
        positions[tetrahedron_indices[tetrahedron, 1]],
        positions[tetrahedron_indices[tetrahedron, 2]],
        positions[tetrahedron_indices[tetrahedron, 3]],
        inverse_rest_matrices[tetrahedron],
    )
    determinant = wp.determinant(deformation)
    if determinant <= 0.0 or not wp.isfinite(determinant):
        wp.atomic_add(invalid_count, 0, 1)
        return
    energy = _neo_hookean_energy(
        deformation,
        shear_moduli[tetrahedron],
        lame_parameters[tetrahedron],
        rest_volumes[tetrahedron],
    )
    wp.atomic_add(output, 0, energy)


@wp.kernel
def _accumulate_neo_hookean_force(
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
    determinant = wp.determinant(deformation)
    if determinant <= 0.0 or not wp.isfinite(determinant):
        return
    gradient = _neo_hookean_gradient(
        deformation,
        shear_moduli[tetrahedron],
        lame_parameters[tetrahedron],
        rest_volumes[tetrahedron],
    )
    for local_vertex in range(4):
        particle = tetrahedron_indices[tetrahedron, local_vertex]
        wp.atomic_sub(forces, particle, gradient * _material_gradient(inverse_rest, local_vertex))


@wp.kernel
def _accumulate_neo_hookean_force_and_hessian(
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
    determinant = wp.determinant(deformation)
    if determinant <= 0.0 or not wp.isfinite(determinant):
        return
    gradient = _neo_hookean_gradient(
        deformation,
        shear_moduli[tetrahedron],
        lame_parameters[tetrahedron],
        rest_volumes[tetrahedron],
    )
    hessian = _project_psd(
        _neo_hookean_hessian(
            deformation,
            shear_moduli[tetrahedron],
            lame_parameters[tetrahedron],
            rest_volumes[tetrahedron],
        )
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


class ConstraintTetrahedronNeoHookean(_TetrahedronElasticConstraintBase):
    """A batch of logarithmic compressible Neo-Hookean tetrahedra."""

    def accumulate_energy(
        self,
        positions: wp.array[wp.vec3],
        output: wp.array[float],
        invalid_count: wp.array[int],
    ) -> None:
        """Add logarithmic Neo-Hookean energy evaluated at ``positions``."""
        self._validate_energy(positions, output, invalid_count)
        wp.launch(
            _accumulate_neo_hookean_energy,
            dim=len(self.rest_volumes),
            inputs=[
                self.tetrahedron_indices,
                self.inverse_rest_matrices,
                self.rest_volumes,
                self.shear_moduli,
                self.lame_parameters,
                positions,
            ],
            outputs=[output, invalid_count],
            device=self.device,
        )

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add logarithmic Neo-Hookean forces evaluated at ``positions``."""
        self._validate_force(positions, output)
        wp.launch(
            _accumulate_neo_hookean_force,
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
        """Add exact forces and PSD-projected Hessian blocks."""
        self._validate_force(positions, force_output)
        self._validate_hessian(hessian_values)
        wp.launch(
            _accumulate_neo_hookean_force_and_hessian,
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
