# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for LIMX kinematic mesh contact."""

import warp as wp


@wp.kernel
def accumulate_contact_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    force = stiffness * depths[contact] * directions[contact]
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * force)


@wp.kernel
def contact_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    direction = directions[contact]
    projected = float(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        projected += weights[contact, local_index] * wp.dot(direction, vector[particle])

    product = stiffness * projected * direction
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * product)


@wp.kernel
def accumulate_contact_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    rank_one = stiffness * wp.outer(directions[contact], directions[contact])
    for local_index in range(arity):
        particle = ids[contact, local_index]
        weight = weights[contact, local_index]
        wp.atomic_add(output, particle, weight * weight * rank_one)


@wp.func
def _relative_velocity(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    rigid_velocities: wp.array[wp.vec3],
    contact: int,
    arity: int,
    particle_velocities: wp.array[wp.vec3],
):
    relative = rigid_velocities[contact]
    for local_index in range(arity):
        relative += weights[contact, local_index] * particle_velocities[ids[contact, local_index]]
    return relative


@wp.kernel
def accumulate_damping_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    damping: float,
    particle_velocities: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    direction = directions[contact]
    normal_velocity = wp.dot(
        direction,
        _relative_velocity(ids, weights, rigid_velocities, contact, arity, particle_velocities),
    )
    if normal_velocity >= 0.0:
        return
    force = -damping * normal_velocity * direction
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * force)


@wp.kernel
def damping_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    damping_over_dt: float,
    particle_velocities: wp.array[wp.vec3],
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    direction = directions[contact]
    normal_velocity = wp.dot(
        direction,
        _relative_velocity(ids, weights, rigid_velocities, contact, arity, particle_velocities),
    )
    if normal_velocity >= 0.0:
        return

    projected = float(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        projected += weights[contact, local_index] * wp.dot(direction, vector[particle])
    product = damping_over_dt * projected * direction
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * product)


@wp.kernel
def accumulate_damping_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    damping_over_dt: float,
    particle_velocities: wp.array[wp.vec3],
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    direction = directions[contact]
    normal_velocity = wp.dot(
        direction,
        _relative_velocity(ids, weights, rigid_velocities, contact, arity, particle_velocities),
    )
    if normal_velocity >= 0.0:
        return

    rank_one = damping_over_dt * wp.outer(direction, direction)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        weight = weights[contact, local_index]
        wp.atomic_add(output, particle, weight * weight * rank_one)


@wp.func
def _friction_force_hessian(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    rigid_velocities: wp.array[wp.vec3],
    contact: int,
    arity: int,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    dt: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
):
    relative_displacement = dt * rigid_velocities[contact]
    for local_index in range(arity):
        particle = ids[contact, local_index]
        relative_displacement += weights[contact, local_index] * (
            positions[particle] - anchor_positions[particle]
        )

    direction = directions[contact]
    tangent_displacement = relative_displacement - direction * wp.dot(direction, relative_displacement)
    tangent_length = wp.length(tangent_displacement)
    normal_load = stiffness * depths[contact]
    if tangent_length <= 0.0 or normal_load <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)

    friction_over_length = float(0.0)
    if tangent_length > displacement_epsilon:
        friction_over_length = 1.0 / tangent_length
    else:
        friction_over_length = (-tangent_length / displacement_epsilon + 2.0) / displacement_epsilon
    scale = friction * normal_load * friction_over_length
    tangent = wp.identity(3, float) - wp.outer(direction, direction)
    return -scale * tangent_displacement, scale * tangent


@wp.kernel
def accumulate_friction_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    rigid_velocities: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    dt: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    force, _hessian = _friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        rigid_velocities,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        dt,
        positions,
        anchor_positions,
    )
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * force)


@wp.kernel
def friction_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    rigid_velocities: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    dt: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    _force, hessian = _friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        rigid_velocities,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        dt,
        positions,
        anchor_positions,
    )
    weighted_vector = wp.vec3(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        weighted_vector += weights[contact, local_index] * vector[particle]
    product = hessian * weighted_vector
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * product)


@wp.kernel
def accumulate_friction_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    rigid_velocities: wp.array[wp.vec3],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    dt: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    _force, hessian = _friction_force_hessian(
        ids,
        weights,
        directions,
        depths,
        rigid_velocities,
        contact,
        arity,
        stiffness,
        friction,
        displacement_epsilon,
        dt,
        positions,
        anchor_positions,
    )
    for local_index in range(arity):
        particle = ids[contact, local_index]
        weight = weights[contact, local_index]
        wp.atomic_add(output, particle, weight * weight * hessian)
