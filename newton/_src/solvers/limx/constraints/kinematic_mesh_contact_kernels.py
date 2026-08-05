# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for LIMX kinematic mesh contact."""

import warp as wp

_MIN_BARYCENTRIC_DENOMINATOR = 1.0e-12
_MIN_CONTACT_DISTANCE = 1.0e-7
_MIN_GEOMETRY_NORM = 1.0e-8


@wp.func
def _triangle_barycentric(
    position_0: wp.vec3,
    position_1: wp.vec3,
    position_2: wp.vec3,
    point: wp.vec3,
):
    edge_0 = position_0 - position_2
    edge_1 = position_1 - position_2
    relative = point - position_2
    dot_00 = wp.dot(edge_0, edge_0)
    dot_01 = wp.dot(edge_0, edge_1)
    dot_02 = wp.dot(edge_0, relative)
    dot_11 = wp.dot(edge_1, edge_1)
    dot_12 = wp.dot(edge_1, relative)
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    if wp.abs(denominator) <= _MIN_BARYCENTRIC_DENOMINATOR:
        return wp.vec3(-1.0)
    inverse = 1.0 / denominator
    barycentric_0 = (dot_11 * dot_02 - dot_01 * dot_12) * inverse
    barycentric_1 = (dot_00 * dot_12 - dot_01 * dot_02) * inverse
    return wp.vec3(barycentric_0, barycentric_1, 1.0 - barycentric_0 - barycentric_1)


@wp.kernel
def update_triangle_bounds(
    positions: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    triangle = wp.tid()
    position_0 = positions[triangles[triangle, 0]]
    position_1 = positions[triangles[triangle, 1]]
    position_2 = positions[triangles[triangle, 2]]
    lower_bounds[triangle] = wp.min(wp.min(position_0, position_1), position_2)
    upper_bounds[triangle] = wp.max(wp.max(position_0, position_1), position_2)


@wp.kernel
def detect_cloth_vertex_rigid_face(
    rigid_triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    rigid_triangles: wp.array2d[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    contact_rigid_velocities: wp.array[wp.vec3],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    vertex = wp.tid()
    vertex_position = cloth_positions[vertex]
    query = wp.bvh_query_aabb(
        rigid_triangle_bvh_id,
        vertex_position - wp.vec3(thickness),
        vertex_position + wp.vec3(thickness),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        index_0 = rigid_triangles[triangle, 0]
        index_1 = rigid_triangles[triangle, 1]
        index_2 = rigid_triangles[triangle, 2]
        position_0 = rigid_positions[index_0]
        position_1 = rigid_positions[index_1]
        position_2 = rigid_positions[index_2]
        normal_raw = wp.cross(position_1 - position_0, position_2 - position_0)
        normal_length = wp.length(normal_raw)
        if normal_length <= _MIN_GEOMETRY_NORM:
            continue
        normal = normal_raw / normal_length
        signed_distance = wp.dot(vertex_position - position_0, normal)
        distance = wp.abs(signed_distance)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue
        projected = vertex_position - signed_distance * normal
        barycentric = _triangle_barycentric(position_0, position_1, position_2, projected)
        if barycentric[0] < 0.0 or barycentric[1] < 0.0 or barycentric[2] < 0.0:
            continue

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue
        if signed_distance < 0.0:
            normal = -normal
        rigid_velocity = (
            barycentric[0] * rigid_velocities[index_0]
            + barycentric[1] * rigid_velocities[index_1]
            + barycentric[2] * rigid_velocities[index_2]
        )
        contact_ids[contact, 0] = vertex
        contact_weights[contact, 0] = 1.0
        contact_directions[contact] = normal
        contact_depths[contact] = thickness - distance
        contact_rigid_velocities[contact] = -rigid_velocity


@wp.kernel
def detect_rigid_vertex_cloth_face(
    cloth_triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    cloth_triangles: wp.array2d[int],
    rigid_positions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    contact_rigid_velocities: wp.array[wp.vec3],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    rigid_vertex = wp.tid()
    rigid_position = rigid_positions[rigid_vertex]
    query = wp.bvh_query_aabb(
        cloth_triangle_bvh_id,
        rigid_position - wp.vec3(thickness),
        rigid_position + wp.vec3(thickness),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        index_0 = cloth_triangles[triangle, 0]
        index_1 = cloth_triangles[triangle, 1]
        index_2 = cloth_triangles[triangle, 2]
        position_0 = cloth_positions[index_0]
        position_1 = cloth_positions[index_1]
        position_2 = cloth_positions[index_2]
        normal_raw = wp.cross(position_1 - position_0, position_2 - position_0)
        normal_length = wp.length(normal_raw)
        if normal_length <= _MIN_GEOMETRY_NORM:
            continue
        normal = normal_raw / normal_length
        signed_distance = wp.dot(rigid_position - position_0, normal)
        distance = wp.abs(signed_distance)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue
        projected = rigid_position - signed_distance * normal
        barycentric = _triangle_barycentric(position_0, position_1, position_2, projected)
        if barycentric[0] < 0.0 or barycentric[1] < 0.0 or barycentric[2] < 0.0:
            continue

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue
        if signed_distance < 0.0:
            normal = -normal
        contact_ids[contact, 0] = index_0
        contact_ids[contact, 1] = index_1
        contact_ids[contact, 2] = index_2
        contact_weights[contact, 0] = -barycentric[0]
        contact_weights[contact, 1] = -barycentric[1]
        contact_weights[contact, 2] = -barycentric[2]
        contact_directions[contact] = normal
        contact_depths[contact] = thickness - distance
        contact_rigid_velocities[contact] = rigid_velocities[rigid_vertex]


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
        relative_displacement += weights[contact, local_index] * (positions[particle] - anchor_positions[particle])

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
