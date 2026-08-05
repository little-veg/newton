# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free cloth contact against kinematic triangle surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ....geometry import GeoType
from ....math import velocity_at_point
from ....sim import Model
from ....utils.mesh import MeshAdjacency


@wp.kernel
def _update_collider_vertices(
    local_positions: wp.array[wp.vec3],
    body_indices: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    world_positions: wp.array[wp.vec3],
    world_velocities: wp.array[wp.vec3],
):
    vertex = wp.tid()
    body = body_indices[vertex]
    local_position = local_positions[vertex]
    if body < 0:
        world_positions[vertex] = local_position
        world_velocities[vertex] = wp.vec3(0.0)
        return

    world_position = wp.transform_point(body_q[body], local_position)
    center_of_mass = wp.transform_point(body_q[body], body_com[body])
    world_positions[vertex] = world_position
    world_velocities[vertex] = velocity_at_point(body_qd[body], world_position - center_of_mass)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    translation = transform[:3]
    vector = transform[3:6]
    scalar = transform[6]
    twice_cross = 2.0 * np.cross(vector, points)
    return points + scalar * twice_cross + np.cross(vector, twice_cross) + translation


def _box_surface(half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = half_extents
    vertices = np.asarray(
        [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ],
        dtype=np.float32,
    )
    triangles = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int32,
    )
    return vertices, triangles


class ConstraintKinematicMeshContact:
    """Matrix-free cloth contact against selected kinematic rigid shapes."""

    def __init__(
        self,
        model: Model,
        shape_indices: Sequence[int],
        thickness: float,
        stiffness: float,
        normal_damping: float,
        friction: float,
        friction_epsilon: float,
        max_contacts: int = 32768,
    ):
        """Create a kinematic triangle-surface contact operator.

        Args:
            model: Model containing the cloth particles and rigid shapes.
            shape_indices: Rigid mesh or box shape indices to triangulate.
            thickness: Contact activation distance [m].
            stiffness: Normal penalty stiffness [N/m].
            normal_damping: Approaching normal damping [N·s/m].
            friction: Coulomb friction coefficient.
            friction_epsilon: Relative-velocity regularization [m/s].
            max_contacts: Capacity of each contact stencil buffer.
        """
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("thickness must be finite and positive")
        if not np.isfinite(stiffness) or stiffness <= 0.0:
            raise ValueError("stiffness must be finite and positive")
        if not np.isfinite(normal_damping) or normal_damping < 0.0:
            raise ValueError("normal_damping must be finite and nonnegative")
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError("friction must be finite and nonnegative")
        if not np.isfinite(friction_epsilon) or friction_epsilon <= 0.0:
            raise ValueError("friction_epsilon must be finite and positive")
        if max_contacts <= 0:
            raise ValueError("max_contacts must be positive")
        if model.particle_count <= 0 or model.tri_count <= 0 or model.tri_indices is None:
            raise ValueError("model must contain a particle triangle mesh")

        selected = np.asarray(shape_indices, dtype=np.int32)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("shape_indices must be a nonempty one-dimensional sequence")
        if np.any(selected < 0) or np.any(selected >= model.shape_count):
            raise ValueError("shape_indices contains an out-of-range shape")

        self.particle_count = model.particle_count
        self.device = model.device
        self.thickness = float(thickness)
        self.stiffness = float(stiffness)
        self.normal_damping = float(normal_damping)
        self.friction = float(friction)
        self.friction_epsilon = float(friction_epsilon)
        self.max_contacts = int(max_contacts)

        shape_types = model.shape_type.numpy()
        shape_scales = model.shape_scale.numpy()
        shape_transforms = model.shape_transform.numpy()
        shape_bodies = model.shape_body.numpy()
        local_positions: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        bodies: list[np.ndarray] = []
        vertex_offset = 0

        for shape_index in selected:
            shape = int(shape_index)
            shape_type = GeoType(int(shape_types[shape]))
            if shape_type in (GeoType.MESH, GeoType.CONVEX_MESH):
                source = model.shape_source[shape]
                if source is None:
                    raise ValueError(f"shape {shape} has no mesh source")
                vertices = np.asarray(source.vertices, dtype=np.float32) * shape_scales[shape]
                shape_triangles = np.asarray(source.indices, dtype=np.int32).reshape((-1, 3))
            elif shape_type == GeoType.BOX:
                vertices, shape_triangles = _box_surface(shape_scales[shape])
            else:
                raise ValueError(f"shape {shape} uses unsupported geometry {shape_type.name}")

            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
                raise ValueError(f"shape {shape} has malformed vertices")
            if shape_triangles.ndim != 2 or shape_triangles.shape[1] != 3 or len(shape_triangles) == 0:
                raise ValueError(f"shape {shape} has malformed triangles")
            if np.any(shape_triangles < 0) or np.any(shape_triangles >= len(vertices)):
                raise ValueError(f"shape {shape} has out-of-range triangle indices")

            transformed = _transform_points(vertices, shape_transforms[shape]).astype(np.float32)
            local_positions.append(transformed)
            triangles.append(shape_triangles + vertex_offset)
            bodies.append(np.full(len(vertices), int(shape_bodies[shape]), dtype=np.int32))
            vertex_offset += len(vertices)

        local_positions_np = np.concatenate(local_positions, axis=0)
        triangles_np = np.concatenate(triangles, axis=0)
        bodies_np = np.concatenate(bodies, axis=0)
        edges_np = MeshAdjacency(triangles_np).edge_indices.astype(np.int32)

        self.collider_local_positions = wp.array(local_positions_np, dtype=wp.vec3, device=self.device)
        self.collider_body = wp.array(bodies_np, dtype=wp.int32, device=self.device)
        self.collider_triangles = wp.array(triangles_np, dtype=wp.int32, device=self.device)
        self.collider_edges = wp.array(edges_np, dtype=wp.int32, device=self.device)
        self.collider_positions = wp.empty_like(self.collider_local_positions)
        self.collider_velocities = wp.empty_like(self.collider_local_positions)
        self._body_com = model.body_com
        self._zero_body_qd = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self.device)

    def update_colliders(
        self,
        body_q: wp.array[wp.transform],
        body_qd: wp.array[wp.spatial_vector] | None = None,
    ) -> None:
        """Update collider world positions and velocities from rigid state.

        Args:
            body_q: Rigid body transforms [m, unitless quaternion].
            body_qd: Optional rigid body spatial velocities [m/s, rad/s].
        """
        if body_q.device != self.device or body_q.dtype != wp.transform:
            raise ValueError(f"body_q must be a transform array on {self.device}")
        velocities = self._zero_body_qd if body_qd is None else body_qd
        if velocities.device != self.device or velocities.dtype != wp.spatial_vector:
            raise ValueError(f"body_qd must be a spatial-vector array on {self.device}")
        if len(body_q) != len(self._body_com) or len(velocities) != len(self._body_com):
            raise ValueError("rigid state arrays must match the model body count")

        wp.launch(
            _update_collider_vertices,
            dim=len(self.collider_local_positions),
            inputs=[
                self.collider_local_positions,
                self.collider_body,
                body_q,
                velocities,
                self._body_com,
            ],
            outputs=[self.collider_positions, self.collider_velocities],
            device=self.device,
        )

