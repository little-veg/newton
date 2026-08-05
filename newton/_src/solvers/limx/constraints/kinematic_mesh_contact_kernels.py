# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for LIMX kinematic mesh contact."""

import warp as wp

_MIN_BARYCENTRIC_DENOMINATOR = 1.0e-12
_MIN_CONTACT_DISTANCE = 1.0e-7
_MIN_GEOMETRY_NORM = 1.0e-8
_EE_MOLLIFIER_THRESHOLD_SCALE = 1.0e-3


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
def predict_positions(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    dt: float,
    predicted: wp.array[wp.vec3],
):
    index = wp.tid()
    predicted[index] = positions[index] + dt * velocities[index]


@wp.kernel
def update_swept_triangle_bounds(
    step_positions: wp.array[wp.vec3],
    predicted_positions: wp.array[wp.vec3],
    current_positions: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    triangle = wp.tid()
    index_0 = triangles[triangle, 0]
    index_1 = triangles[triangle, 1]
    index_2 = triangles[triangle, 2]
    lower = wp.min(
        wp.min(step_positions[index_0], step_positions[index_1]),
        step_positions[index_2],
    )
    upper = wp.max(
        wp.max(step_positions[index_0], step_positions[index_1]),
        step_positions[index_2],
    )
    lower = wp.min(
        lower,
        wp.min(
            wp.min(predicted_positions[index_0], predicted_positions[index_1]),
            predicted_positions[index_2],
        ),
    )
    upper = wp.max(
        upper,
        wp.max(
            wp.max(predicted_positions[index_0], predicted_positions[index_1]),
            predicted_positions[index_2],
        ),
    )
    lower = wp.min(
        lower,
        wp.min(
            wp.min(current_positions[index_0], current_positions[index_1]),
            current_positions[index_2],
        ),
    )
    upper = wp.max(
        upper,
        wp.max(
            wp.max(current_positions[index_0], current_positions[index_1]),
            current_positions[index_2],
        ),
    )
    lower_bounds[triangle] = lower
    upper_bounds[triangle] = upper


@wp.kernel
def update_edge_bounds(
    positions: wp.array[wp.vec3],
    edges: wp.array2d[int],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    edge = wp.tid()
    position_0 = positions[edges[edge, 2]]
    position_1 = positions[edges[edge, 3]]
    lower_bounds[edge] = wp.min(position_0, position_1)
    upper_bounds[edge] = wp.max(position_0, position_1)


@wp.kernel
def update_swept_edge_bounds(
    step_positions: wp.array[wp.vec3],
    predicted_positions: wp.array[wp.vec3],
    current_positions: wp.array[wp.vec3],
    edges: wp.array2d[int],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    edge = wp.tid()
    index_0 = edges[edge, 2]
    index_1 = edges[edge, 3]
    lower = wp.min(step_positions[index_0], step_positions[index_1])
    upper = wp.max(step_positions[index_0], step_positions[index_1])
    lower = wp.min(lower, wp.min(predicted_positions[index_0], predicted_positions[index_1]))
    upper = wp.max(upper, wp.max(predicted_positions[index_0], predicted_positions[index_1]))
    lower = wp.min(lower, wp.min(current_positions[index_0], current_positions[index_1]))
    upper = wp.max(upper, wp.max(current_positions[index_0], current_positions[index_1]))
    lower_bounds[edge] = lower
    upper_bounds[edge] = upper


@wp.kernel
def detect_cloth_vertex_rigid_face(
    rigid_triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    cloth_step_positions: wp.array[wp.vec3],
    cloth_predicted_positions: wp.array[wp.vec3],
    rigid_positions: wp.array[wp.vec3],
    rigid_step_positions: wp.array[wp.vec3],
    rigid_predicted_positions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    rigid_triangles: wp.array2d[int],
    rigid_triangle_one_sided: wp.array[int],
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
    vertex_step_position = cloth_step_positions[vertex]
    vertex_predicted_position = cloth_predicted_positions[vertex]
    query = wp.bvh_query_aabb(
        rigid_triangle_bvh_id,
        wp.min(wp.min(vertex_position, vertex_step_position), vertex_predicted_position) - wp.vec3(thickness),
        wp.max(wp.max(vertex_position, vertex_step_position), vertex_predicted_position) + wp.vec3(thickness),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        index_0 = rigid_triangles[triangle, 0]
        index_1 = rigid_triangles[triangle, 1]
        index_2 = rigid_triangles[triangle, 2]
        position_0 = rigid_positions[index_0]
        position_1 = rigid_positions[index_1]
        position_2 = rigid_positions[index_2]
        step_position_0 = rigid_step_positions[index_0]
        step_position_1 = rigid_step_positions[index_1]
        step_position_2 = rigid_step_positions[index_2]
        predicted_position_0 = rigid_predicted_positions[index_0]
        predicted_position_1 = rigid_predicted_positions[index_1]
        predicted_position_2 = rigid_predicted_positions[index_2]
        normal_raw = wp.cross(position_1 - position_0, position_2 - position_0)
        step_normal_raw = wp.cross(step_position_1 - step_position_0, step_position_2 - step_position_0)
        normal_length = wp.length(normal_raw)
        step_normal_length = wp.length(step_normal_raw)
        if normal_length <= _MIN_GEOMETRY_NORM or step_normal_length <= _MIN_GEOMETRY_NORM:
            continue
        normal = normal_raw / normal_length
        step_normal = step_normal_raw / step_normal_length
        if wp.dot(normal, step_normal) < 0.0:
            normal = -normal
        if rigid_triangle_one_sided[triangle] == 0:
            step_signed_distance = wp.dot(vertex_step_position - step_position_0, step_normal)
            if step_signed_distance < 0.0:
                normal = -normal
        signed_gap = wp.dot(vertex_position - position_0, normal)
        predicted_signed_gap = wp.dot(vertex_predicted_position - predicted_position_0, normal)
        if signed_gap >= thickness and predicted_signed_gap >= thickness:
            continue
        projected = vertex_position - signed_gap * normal
        barycentric = _triangle_barycentric(position_0, position_1, position_2, projected)
        predicted_projected = vertex_predicted_position - predicted_signed_gap * normal
        predicted_barycentric = _triangle_barycentric(
            predicted_position_0,
            predicted_position_1,
            predicted_position_2,
            predicted_projected,
        )
        current_inside = barycentric[0] >= 0.0 and barycentric[1] >= 0.0 and barycentric[2] >= 0.0
        predicted_inside = (
            predicted_barycentric[0] >= 0.0 and predicted_barycentric[1] >= 0.0 and predicted_barycentric[2] >= 0.0
        )
        if not current_inside and not predicted_inside:
            continue
        if not current_inside:
            barycentric = predicted_barycentric

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue
        rigid_velocity = (
            barycentric[0] * rigid_velocities[index_0]
            + barycentric[1] * rigid_velocities[index_1]
            + barycentric[2] * rigid_velocities[index_2]
        )
        contact_ids[contact, 0] = vertex
        contact_weights[contact, 0] = 1.0
        contact_directions[contact] = normal
        contact_depths[contact] = thickness - signed_gap
        contact_rigid_velocities[contact] = -rigid_velocity


@wp.kernel
def detect_rigid_vertex_cloth_face(
    cloth_triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    cloth_step_positions: wp.array[wp.vec3],
    cloth_predicted_positions: wp.array[wp.vec3],
    cloth_triangles: wp.array2d[int],
    rigid_positions: wp.array[wp.vec3],
    rigid_step_positions: wp.array[wp.vec3],
    rigid_predicted_positions: wp.array[wp.vec3],
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
    rigid_step_position = rigid_step_positions[rigid_vertex]
    rigid_predicted_position = rigid_predicted_positions[rigid_vertex]
    query = wp.bvh_query_aabb(
        cloth_triangle_bvh_id,
        wp.min(wp.min(rigid_position, rigid_step_position), rigid_predicted_position) - wp.vec3(thickness),
        wp.max(wp.max(rigid_position, rigid_step_position), rigid_predicted_position) + wp.vec3(thickness),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        index_0 = cloth_triangles[triangle, 0]
        index_1 = cloth_triangles[triangle, 1]
        index_2 = cloth_triangles[triangle, 2]
        position_0 = cloth_positions[index_0]
        position_1 = cloth_positions[index_1]
        position_2 = cloth_positions[index_2]
        step_position_0 = cloth_step_positions[index_0]
        step_position_1 = cloth_step_positions[index_1]
        step_position_2 = cloth_step_positions[index_2]
        predicted_position_0 = cloth_predicted_positions[index_0]
        predicted_position_1 = cloth_predicted_positions[index_1]
        predicted_position_2 = cloth_predicted_positions[index_2]
        normal_raw = wp.cross(position_1 - position_0, position_2 - position_0)
        step_normal_raw = wp.cross(step_position_1 - step_position_0, step_position_2 - step_position_0)
        normal_length = wp.length(normal_raw)
        step_normal_length = wp.length(step_normal_raw)
        if normal_length <= _MIN_GEOMETRY_NORM or step_normal_length <= _MIN_GEOMETRY_NORM:
            continue
        normal = normal_raw / normal_length
        step_normal = step_normal_raw / step_normal_length
        if wp.dot(normal, step_normal) < 0.0:
            normal = -normal
        step_signed_distance = wp.dot(rigid_step_position - step_position_0, step_normal)
        if step_signed_distance < 0.0:
            normal = -normal
        signed_gap = wp.dot(rigid_position - position_0, normal)
        predicted_signed_gap = wp.dot(rigid_predicted_position - predicted_position_0, normal)
        if signed_gap >= thickness and predicted_signed_gap >= thickness:
            continue
        projected = rigid_position - signed_gap * normal
        barycentric = _triangle_barycentric(position_0, position_1, position_2, projected)
        predicted_projected = rigid_predicted_position - predicted_signed_gap * normal
        predicted_barycentric = _triangle_barycentric(
            predicted_position_0,
            predicted_position_1,
            predicted_position_2,
            predicted_projected,
        )
        current_inside = barycentric[0] >= 0.0 and barycentric[1] >= 0.0 and barycentric[2] >= 0.0
        predicted_inside = (
            predicted_barycentric[0] >= 0.0 and predicted_barycentric[1] >= 0.0 and predicted_barycentric[2] >= 0.0
        )
        if not current_inside and not predicted_inside:
            continue
        if not current_inside:
            barycentric = predicted_barycentric

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue
        contact_ids[contact, 0] = index_0
        contact_ids[contact, 1] = index_1
        contact_ids[contact, 2] = index_2
        contact_weights[contact, 0] = -barycentric[0]
        contact_weights[contact, 1] = -barycentric[1]
        contact_weights[contact, 2] = -barycentric[2]
        contact_directions[contact] = normal
        contact_depths[contact] = thickness - signed_gap
        contact_rigid_velocities[contact] = rigid_velocities[rigid_vertex]


@wp.kernel
def detect_cloth_edge_rigid_edge(
    rigid_edge_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    cloth_positions: wp.array[wp.vec3],
    cloth_step_positions: wp.array[wp.vec3],
    cloth_predicted_positions: wp.array[wp.vec3],
    cloth_rest_positions: wp.array[wp.vec3],
    cloth_edges: wp.array2d[int],
    rigid_positions: wp.array[wp.vec3],
    rigid_step_positions: wp.array[wp.vec3],
    rigid_predicted_positions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    rigid_edges: wp.array2d[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    contact_rigid_velocities: wp.array[wp.vec3],
    contact_mollifier_thresholds: wp.array[float],
    contact_rigid_edge_vectors: wp.array[wp.vec3],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    cloth_edge = wp.tid()
    cloth_index_0 = cloth_edges[cloth_edge, 2]
    cloth_index_1 = cloth_edges[cloth_edge, 3]
    cloth_position_0 = cloth_positions[cloth_index_0]
    cloth_position_1 = cloth_positions[cloth_index_1]
    cloth_step_position_0 = cloth_step_positions[cloth_index_0]
    cloth_step_position_1 = cloth_step_positions[cloth_index_1]
    cloth_predicted_position_0 = cloth_predicted_positions[cloth_index_0]
    cloth_predicted_position_1 = cloth_predicted_positions[cloth_index_1]
    query = wp.bvh_query_aabb(
        rigid_edge_bvh_id,
        wp.min(
            wp.min(cloth_position_0, cloth_position_1),
            wp.min(
                wp.min(cloth_step_position_0, cloth_step_position_1),
                wp.min(cloth_predicted_position_0, cloth_predicted_position_1),
            ),
        )
        - wp.vec3(thickness),
        wp.max(
            wp.max(cloth_position_0, cloth_position_1),
            wp.max(
                wp.max(cloth_step_position_0, cloth_step_position_1),
                wp.max(cloth_predicted_position_0, cloth_predicted_position_1),
            ),
        )
        + wp.vec3(thickness),
    )
    rigid_edge = wp.int32(-1)
    while wp.bvh_query_next(query, rigid_edge):
        rigid_index_0 = rigid_edges[rigid_edge, 2]
        rigid_index_1 = rigid_edges[rigid_edge, 3]
        rigid_position_0 = rigid_positions[rigid_index_0]
        rigid_position_1 = rigid_positions[rigid_index_1]
        rigid_step_position_0 = rigid_step_positions[rigid_index_0]
        rigid_step_position_1 = rigid_step_positions[rigid_index_1]
        rigid_predicted_position_0 = rigid_predicted_positions[rigid_index_0]
        rigid_predicted_position_1 = rigid_predicted_positions[rigid_index_1]
        parameters = wp.closest_point_edge_edge(
            cloth_position_0,
            cloth_position_1,
            rigid_position_0,
            rigid_position_1,
            1.0e-5,
        )
        cloth_parameter = parameters[0]
        rigid_parameter = parameters[1]
        current_interior = not (
            cloth_parameter <= _MIN_CONTACT_DISTANCE
            or cloth_parameter >= 1.0 - _MIN_CONTACT_DISTANCE
            or rigid_parameter <= _MIN_CONTACT_DISTANCE
            or rigid_parameter >= 1.0 - _MIN_CONTACT_DISTANCE
        )
        predicted_parameters = wp.closest_point_edge_edge(
            cloth_predicted_position_0,
            cloth_predicted_position_1,
            rigid_predicted_position_0,
            rigid_predicted_position_1,
            1.0e-5,
        )
        predicted_cloth_parameter = predicted_parameters[0]
        predicted_rigid_parameter = predicted_parameters[1]
        predicted_interior = not (
            predicted_cloth_parameter <= _MIN_CONTACT_DISTANCE
            or predicted_cloth_parameter >= 1.0 - _MIN_CONTACT_DISTANCE
            or predicted_rigid_parameter <= _MIN_CONTACT_DISTANCE
            or predicted_rigid_parameter >= 1.0 - _MIN_CONTACT_DISTANCE
        )
        if not current_interior and not predicted_interior:
            continue
        if not current_interior:
            cloth_parameter = predicted_cloth_parameter
            rigid_parameter = predicted_rigid_parameter
        cloth_closest = wp.lerp(cloth_position_0, cloth_position_1, cloth_parameter)
        rigid_closest = wp.lerp(rigid_position_0, rigid_position_1, rigid_parameter)
        separation = cloth_closest - rigid_closest
        distance = wp.length(separation)
        step_cloth_closest = wp.lerp(cloth_step_position_0, cloth_step_position_1, cloth_parameter)
        step_rigid_closest = wp.lerp(rigid_step_position_0, rigid_step_position_1, rigid_parameter)
        step_separation = step_cloth_closest - step_rigid_closest
        step_distance = wp.length(step_separation)
        if step_distance <= _MIN_CONTACT_DISTANCE:
            continue
        reference_direction = step_separation / step_distance
        direction = reference_direction
        signed_gap = wp.dot(separation, reference_direction)
        if distance > _MIN_CONTACT_DISTANCE:
            direction = separation / distance
            if wp.dot(direction, reference_direction) < 0.0:
                direction = -direction
            signed_gap = wp.dot(separation, direction)
        predicted_cloth_closest = wp.lerp(
            cloth_predicted_position_0,
            cloth_predicted_position_1,
            predicted_cloth_parameter,
        )
        predicted_rigid_closest = wp.lerp(
            rigid_predicted_position_0,
            rigid_predicted_position_1,
            predicted_rigid_parameter,
        )
        predicted_signed_gap = wp.dot(predicted_cloth_closest - predicted_rigid_closest, reference_direction)
        if signed_gap >= thickness and predicted_signed_gap >= thickness:
            continue

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue
        contact_ids[contact, 0] = cloth_index_0
        contact_ids[contact, 1] = cloth_index_1
        contact_weights[contact, 0] = 1.0 - cloth_parameter
        contact_weights[contact, 1] = cloth_parameter
        contact_directions[contact] = direction
        contact_depths[contact] = thickness - signed_gap
        rigid_velocity = wp.lerp(
            rigid_velocities[rigid_index_0],
            rigid_velocities[rigid_index_1],
            rigid_parameter,
        )
        contact_rigid_velocities[contact] = -rigid_velocity
        cloth_rest_edge = cloth_rest_positions[cloth_index_1] - cloth_rest_positions[cloth_index_0]
        rigid_current_edge = rigid_position_1 - rigid_position_0
        contact_rigid_edge_vectors[contact] = rigid_current_edge
        contact_mollifier_thresholds[contact] = (
            _EE_MOLLIFIER_THRESHOLD_SCALE
            * wp.dot(cloth_rest_edge, cloth_rest_edge)
            * wp.dot(rigid_current_edge, rigid_current_edge)
        )


@wp.func
def _edge_edge_mollified_residual_data(
    cloth_edge: wp.vec3,
    rigid_edge: wp.vec3,
    threshold: float,
):
    cross_product = wp.cross(cloth_edge, rigid_edge)
    root = wp.sqrt(wp.max(2.0 * threshold - wp.dot(cross_product, cross_product), threshold))
    scale = root / threshold
    scale_gradient = -0.5 / (threshold * root)
    return cross_product, scale, scale_gradient


@wp.func
def _edge_edge_mollified_residual_jacobian_multiply(
    cloth_edge: wp.vec3,
    rigid_edge: wp.vec3,
    depth: float,
    threshold: float,
    cloth_edge_delta: wp.vec3,
    depth_delta: float,
):
    cross_product, scale, scale_gradient = _edge_edge_mollified_residual_data(
        cloth_edge,
        rigid_edge,
        threshold,
    )
    cross_delta = wp.cross(cloth_edge_delta, rigid_edge)
    cross_squared_delta = 2.0 * wp.dot(cross_product, cross_delta)
    return (
        depth * scale * cross_delta
        + (depth * scale_gradient * cross_squared_delta + depth_delta * scale) * cross_product
    )


@wp.func
def _edge_edge_mollified_residual_jacobian_transpose_multiply(
    cloth_edge: wp.vec3,
    rigid_edge: wp.vec3,
    depth: float,
    threshold: float,
    residual_vector: wp.vec3,
):
    cross_product, scale, scale_gradient = _edge_edge_mollified_residual_data(
        cloth_edge,
        rigid_edge,
        threshold,
    )
    cross_projection = wp.dot(cross_product, residual_vector)
    cross_squared_product = depth * scale_gradient * cross_projection
    cloth_edge_product = depth * scale * wp.cross(rigid_edge, residual_vector)
    cloth_edge_product += cross_squared_product * 2.0 * wp.cross(rigid_edge, cross_product)
    depth_product = scale * cross_projection
    return cloth_edge_product, depth_product


@wp.kernel
def prepare_edge_edge_mollifier(
    ids: wp.array2d[int],
    rigid_edge_vectors: wp.array[wp.vec3],
    thresholds: wp.array[float],
    count: wp.array[int],
    capacity: int,
    positions: wp.array[wp.vec3],
    mollifier_active: wp.array[int],
    load_scales: wp.array[float],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return

    cloth_edge = positions[ids[contact, 1]] - positions[ids[contact, 0]]
    cross_product = wp.cross(cloth_edge, rigid_edge_vectors[contact])
    threshold = thresholds[contact]
    cross_squared = wp.dot(cross_product, cross_product)
    active = threshold > _MIN_GEOMETRY_NORM * _MIN_GEOMETRY_NORM and cross_squared < threshold
    mollifier_active[contact] = wp.int32(active)
    load_scales[contact] = 1.0
    if active:
        load_scales[contact] = wp.clamp(
            cross_squared * (2.0 * threshold - cross_squared) / (threshold * threshold),
            0.0,
            1.0,
        )


@wp.func
def _kinematic_edge_edge_gauss_newton_multiply(
    cloth_edge: wp.vec3,
    rigid_edge: wp.vec3,
    weights: wp.vec2,
    direction: wp.vec3,
    depth: float,
    threshold: float,
    vector_0: wp.vec3,
    vector_1: wp.vec3,
):
    depth_delta = -wp.dot(direction, weights[0] * vector_0 + weights[1] * vector_1)
    residual_product = _edge_edge_mollified_residual_jacobian_multiply(
        cloth_edge,
        rigid_edge,
        depth,
        threshold,
        vector_1 - vector_0,
        depth_delta,
    )
    cloth_edge_product, depth_product = _edge_edge_mollified_residual_jacobian_transpose_multiply(
        cloth_edge,
        rigid_edge,
        depth,
        threshold,
        residual_product,
    )
    return (
        -cloth_edge_product - weights[0] * depth_product * direction,
        cloth_edge_product - weights[1] * depth_product * direction,
    )


@wp.func
def _kinematic_edge_edge_gauss_newton_diagonal_block(
    cloth_edge: wp.vec3,
    rigid_edge: wp.vec3,
    weight: float,
    direction: wp.vec3,
    depth: float,
    threshold: float,
    local_index: int,
):
    columns = wp.mat33(0.0)
    for axis in range(3):
        basis = wp.vec3(0.0)
        basis[axis] = 1.0
        cloth_edge_delta = basis
        if local_index == 0:
            cloth_edge_delta = -basis
        residual_product = _edge_edge_mollified_residual_jacobian_multiply(
            cloth_edge,
            rigid_edge,
            depth,
            threshold,
            cloth_edge_delta,
            -weight * direction[axis],
        )
        cloth_edge_product, depth_product = _edge_edge_mollified_residual_jacobian_transpose_multiply(
            cloth_edge,
            rigid_edge,
            depth,
            threshold,
            residual_product,
        )
        local_product = cloth_edge_product
        if local_index == 0:
            local_product = -cloth_edge_product
        local_product -= weight * depth_product * direction
        columns[0, axis] = local_product[0]
        columns[1, axis] = local_product[1]
        columns[2, axis] = local_product[2]
    return columns


@wp.kernel
def accumulate_mollified_edge_edge_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    rigid_edge_vectors: wp.array[wp.vec3],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    positions: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    if depths[contact] <= 0.0:
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    direction = directions[contact]
    depth = depths[contact]
    if mollifier_active[contact] != 0:
        cloth_edge = positions[index_1] - positions[index_0]
        rigid_edge = rigid_edge_vectors[contact]
        threshold = mollifier_thresholds[contact]
        cross_product, residual_scale, _scale_gradient = _edge_edge_mollified_residual_data(
            cloth_edge,
            rigid_edge,
            threshold,
        )
        cloth_edge_product, depth_product = _edge_edge_mollified_residual_jacobian_transpose_multiply(
            cloth_edge,
            rigid_edge,
            depth,
            threshold,
            depth * residual_scale * cross_product,
        )
        gradient_0 = -cloth_edge_product - weights[contact, 0] * depth_product * direction
        gradient_1 = cloth_edge_product - weights[contact, 1] * depth_product * direction
        wp.atomic_add(output, index_0, -stiffness * gradient_0)
        wp.atomic_add(output, index_1, -stiffness * gradient_1)
        return

    force = stiffness * depth * direction
    wp.atomic_add(output, index_0, weights[contact, 0] * force)
    wp.atomic_add(output, index_1, weights[contact, 1] * force)


@wp.kernel
def mollified_edge_edge_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    rigid_edge_vectors: wp.array[wp.vec3],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    positions: wp.array[wp.vec3],
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    if depths[contact] <= 0.0:
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    direction = directions[contact]
    if mollifier_active[contact] != 0:
        product_0, product_1 = _kinematic_edge_edge_gauss_newton_multiply(
            positions[index_1] - positions[index_0],
            rigid_edge_vectors[contact],
            wp.vec2(weights[contact, 0], weights[contact, 1]),
            direction,
            depths[contact],
            mollifier_thresholds[contact],
            vector[index_0],
            vector[index_1],
        )
        wp.atomic_add(output, index_0, stiffness * product_0)
        wp.atomic_add(output, index_1, stiffness * product_1)
        return

    projected = weights[contact, 0] * wp.dot(direction, vector[index_0])
    projected += weights[contact, 1] * wp.dot(direction, vector[index_1])
    product = stiffness * projected * direction
    wp.atomic_add(output, index_0, weights[contact, 0] * product)
    wp.atomic_add(output, index_1, weights[contact, 1] * product)


@wp.kernel
def accumulate_mollified_edge_edge_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    rigid_edge_vectors: wp.array[wp.vec3],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    count: wp.array[int],
    capacity: int,
    stiffness: float,
    positions: wp.array[wp.vec3],
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    if depths[contact] <= 0.0:
        return

    index_0 = ids[contact, 0]
    index_1 = ids[contact, 1]
    direction = directions[contact]
    if mollifier_active[contact] != 0:
        cloth_edge = positions[index_1] - positions[index_0]
        for local_index in range(2):
            block = _kinematic_edge_edge_gauss_newton_diagonal_block(
                cloth_edge,
                rigid_edge_vectors[contact],
                weights[contact, local_index],
                direction,
                depths[contact],
                mollifier_thresholds[contact],
                local_index,
            )
            wp.atomic_add(output, ids[contact, local_index], stiffness * block)
        return

    rank_one = stiffness * wp.outer(direction, direction)
    for local_index in range(2):
        weight = weights[contact, local_index]
        wp.atomic_add(output, ids[contact, local_index], weight * weight * rank_one)


@wp.kernel
def accumulate_contact_force(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    load_scales: wp.array[float],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    output: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    if depths[contact] <= 0.0:
        return

    force = stiffness * depths[contact] * load_scales[contact] * directions[contact]
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * force)


@wp.kernel
def contact_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    load_scales: wp.array[float],
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
    if depths[contact] <= 0.0:
        return

    direction = directions[contact]
    projected = float(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        projected += weights[contact, local_index] * wp.dot(direction, vector[particle])

    product = stiffness * load_scales[contact] * projected * direction
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * product)


@wp.kernel
def accumulate_contact_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    depths: wp.array[float],
    load_scales: wp.array[float],
    count: wp.array[int],
    arity: int,
    capacity: int,
    stiffness: float,
    output: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= wp.min(count[0], capacity):
        return
    if depths[contact] <= 0.0:
        return

    rank_one = stiffness * load_scales[contact] * wp.outer(directions[contact], directions[contact])
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
    load_scales: wp.array[float],
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
    force = -damping * load_scales[contact] * normal_velocity * direction
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * force)


@wp.kernel
def damping_hessian_multiply(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    load_scales: wp.array[float],
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
    product = damping_over_dt * load_scales[contact] * projected * direction
    for local_index in range(arity):
        particle = ids[contact, local_index]
        wp.atomic_add(output, particle, weights[contact, local_index] * product)


@wp.kernel
def accumulate_damping_diagonal(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    directions: wp.array[wp.vec3],
    rigid_velocities: wp.array[wp.vec3],
    load_scales: wp.array[float],
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

    rank_one = damping_over_dt * load_scales[contact] * wp.outer(direction, direction)
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
    load_scales: wp.array[float],
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
    normal_load = stiffness * depths[contact] * load_scales[contact]
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
    load_scales: wp.array[float],
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
        load_scales,
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
    load_scales: wp.array[float],
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
        load_scales,
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
    load_scales: wp.array[float],
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
        load_scales,
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
