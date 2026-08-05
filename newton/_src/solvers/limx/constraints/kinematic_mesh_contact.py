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
from .kinematic_mesh_contact_kernels import (
    accumulate_contact_diagonal,
    accumulate_contact_force,
    accumulate_damping_diagonal,
    accumulate_damping_force,
    accumulate_friction_diagonal,
    accumulate_friction_force,
    accumulate_mollified_edge_edge_diagonal,
    accumulate_mollified_edge_edge_force,
    contact_hessian_multiply,
    damping_hessian_multiply,
    detect_cloth_edge_rigid_edge,
    detect_cloth_vertex_rigid_face,
    detect_rigid_vertex_cloth_face,
    friction_hessian_multiply,
    mollified_edge_edge_hessian_multiply,
    predict_positions,
    prepare_edge_edge_mollifier,
    update_edge_bounds,
    update_swept_edge_bounds,
    update_swept_triangle_bounds,
    update_triangle_bounds,
)


class _KinematicContactBuffer:
    """Fixed-capacity dynamic-only frozen contact data."""

    def __init__(self, arity: int, capacity: int, particle_count: int, device: Any):
        if arity not in (1, 2, 3):
            raise ValueError("contact arity must be one, two, or three")
        if capacity <= 0 or particle_count <= 0:
            raise ValueError("contact capacity and particle count must be positive")
        self.arity = int(arity)
        self.capacity = int(capacity)
        self.particle_count = int(particle_count)
        self.device = wp.get_device(device)
        self.ids = wp.zeros((capacity, arity), dtype=wp.int32, device=self.device)
        self.weights = wp.zeros((capacity, arity), dtype=wp.float32, device=self.device)
        self.directions = wp.zeros(capacity, dtype=wp.vec3, device=self.device)
        self.depths = wp.zeros(capacity, dtype=wp.float32, device=self.device)
        self.load_scales = wp.ones(capacity, dtype=wp.float32, device=self.device)
        self.rigid_velocities = wp.zeros(capacity, dtype=wp.vec3, device=self.device)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.overflow_count = wp.zeros(1, dtype=wp.int32, device=self.device)

    def clear(self) -> None:
        """Reset attempted-contact and overflow counters."""
        self.count.zero_()
        self.overflow_count.zero_()

    def accumulate_force(self, stiffness: float, output: wp.array[wp.vec3]) -> None:
        """Add frozen normal contact forces."""
        self._validate_particle_array(output, wp.vec3)
        wp.launch(
            accumulate_contact_force,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.load_scales,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
            ],
            outputs=[output],
            device=self.device,
        )

    def hessian_multiply(
        self,
        stiffness: float,
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add frozen normal contact Hessian-vector products."""
        self._validate_particle_array(vector, wp.vec3)
        self._validate_particle_array(output, wp.vec3)
        wp.launch(
            contact_hessian_multiply,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.load_scales,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_diagonal(self, stiffness: float, output: wp.array[wp.mat33]) -> None:
        """Add exact diagonal blocks of the frozen normal Hessian."""
        self._validate_particle_array(output, wp.mat33)
        wp.launch(
            accumulate_contact_diagonal,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.load_scales,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_damping_force(
        self,
        damping: float,
        dt: float,
        velocities: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add approaching-only normal damping forces."""
        self._validate_particle_array(velocities, wp.vec3)
        self._validate_particle_array(output, wp.vec3)
        wp.launch(
            accumulate_damping_force,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.rigid_velocities,
                self.load_scales,
                self.count,
                self.arity,
                self.capacity,
                damping,
                velocities,
            ],
            outputs=[output],
            device=self.device,
        )

    def damping_hessian_multiply(
        self,
        damping: float,
        dt: float,
        velocities: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add active normal damping Hessian-vector products."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._validate_particle_array(velocities, wp.vec3)
        self._validate_particle_array(vector, wp.vec3)
        self._validate_particle_array(output, wp.vec3)
        wp.launch(
            damping_hessian_multiply,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.rigid_velocities,
                self.load_scales,
                self.count,
                self.arity,
                self.capacity,
                damping / dt,
                velocities,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_damping_diagonal(
        self,
        damping: float,
        dt: float,
        velocities: wp.array[wp.vec3],
        output: wp.array[wp.mat33],
    ) -> None:
        """Add active normal damping diagonal Hessian blocks."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._validate_particle_array(velocities, wp.vec3)
        self._validate_particle_array(output, wp.mat33)
        wp.launch(
            accumulate_damping_diagonal,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.rigid_velocities,
                self.load_scales,
                self.count,
                self.arity,
                self.capacity,
                damping / dt,
                velocities,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_friction_force(
        self,
        stiffness: float,
        friction: float,
        friction_epsilon: float,
        dt: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add regularized Coulomb friction forces."""
        self._validate_friction_arrays(positions, anchor_positions, output, wp.vec3)
        wp.launch(
            accumulate_friction_force,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.load_scales,
                self.rigid_velocities,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
                friction,
                friction_epsilon * dt,
                dt,
                positions,
                anchor_positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def friction_hessian_multiply(
        self,
        stiffness: float,
        friction: float,
        friction_epsilon: float,
        dt: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add regularized friction Hessian-vector products."""
        self._validate_friction_arrays(positions, anchor_positions, vector, wp.vec3)
        self._validate_particle_array(output, wp.vec3)
        wp.launch(
            friction_hessian_multiply,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.load_scales,
                self.rigid_velocities,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
                friction,
                friction_epsilon * dt,
                dt,
                positions,
                anchor_positions,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_friction_diagonal(
        self,
        stiffness: float,
        friction: float,
        friction_epsilon: float,
        dt: float,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        output: wp.array[wp.mat33],
    ) -> None:
        """Add regularized friction diagonal Hessian blocks."""
        self._validate_friction_arrays(positions, anchor_positions, output, wp.mat33)
        wp.launch(
            accumulate_friction_diagonal,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.load_scales,
                self.rigid_velocities,
                self.count,
                self.arity,
                self.capacity,
                stiffness,
                friction,
                friction_epsilon * dt,
                dt,
                positions,
                anchor_positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def _validate_friction_arrays(
        self,
        positions: wp.array[wp.vec3],
        anchor_positions: wp.array[wp.vec3],
        output: wp.array,
        output_dtype: Any,
    ) -> None:
        self._validate_particle_array(positions, wp.vec3)
        self._validate_particle_array(anchor_positions, wp.vec3)
        self._validate_particle_array(output, output_dtype)

    def _validate_particle_array(self, array: wp.array, dtype: Any) -> None:
        if array.device != self.device or array.dtype != dtype or len(array) != self.particle_count:
            raise ValueError(f"particle array must contain {self.particle_count} {dtype} values on {self.device}")


class _KinematicEdgeEdgeContactBuffer(_KinematicContactBuffer):
    """Dynamic-static EE contacts with stored IPC mollifier thresholds."""

    def __init__(self, capacity: int, particle_count: int, device: Any):
        super().__init__(arity=2, capacity=capacity, particle_count=particle_count, device=device)
        self.mollifier_thresholds = wp.zeros(capacity, dtype=wp.float32, device=self.device)
        self.mollifier_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.rigid_edge_vectors = wp.zeros(capacity, dtype=wp.vec3, device=self.device)

    def prepare_hessian(self, positions: wp.array[wp.vec3]) -> None:
        """Mark near-parallel EE contacts whose IPC mollifier is active."""
        self._validate_particle_array(positions, wp.vec3)
        wp.launch(
            prepare_edge_edge_mollifier,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.rigid_edge_vectors,
                self.mollifier_thresholds,
                self.count,
                self.capacity,
                positions,
            ],
            outputs=[self.mollifier_active, self.load_scales],
            device=self.device,
        )

    def accumulate_force(
        self,
        stiffness: float,
        positions: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add exact forces of the IPC-mollified dynamic-static EE energy."""
        self._validate_particle_array(positions, wp.vec3)
        self._validate_particle_array(output, wp.vec3)
        wp.launch(
            accumulate_mollified_edge_edge_force,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.rigid_edge_vectors,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                stiffness,
                positions,
            ],
            outputs=[output],
            device=self.device,
        )

    def hessian_multiply(
        self,
        stiffness: float,
        positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add Gauss-Newton products of the mollified EE energy."""
        self._validate_particle_array(positions, wp.vec3)
        self._validate_particle_array(vector, wp.vec3)
        self._validate_particle_array(output, wp.vec3)
        wp.launch(
            mollified_edge_edge_hessian_multiply,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.rigid_edge_vectors,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                stiffness,
                positions,
                vector,
            ],
            outputs=[output],
            device=self.device,
        )

    def accumulate_diagonal(
        self,
        stiffness: float,
        positions: wp.array[wp.vec3],
        output: wp.array[wp.mat33],
    ) -> None:
        """Add exact diagonal blocks of the mollified EE Gauss-Newton operator."""
        self._validate_particle_array(positions, wp.vec3)
        self._validate_particle_array(output, wp.mat33)
        wp.launch(
            accumulate_mollified_edge_edge_diagonal,
            dim=self.capacity,
            inputs=[
                self.ids,
                self.weights,
                self.directions,
                self.depths,
                self.rigid_edge_vectors,
                self.mollifier_thresholds,
                self.mollifier_active,
                self.count,
                self.capacity,
                stiffness,
                positions,
            ],
            outputs=[output],
            device=self.device,
        )


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
    """Matrix-free cloth contact against selected kinematic rigid shapes.

    Mesh, convex-mesh, and box shapes are converted to one triangle surface.
    Contact detection includes both vertex-face directions and edge-edge
    pairs. Forces and matrix-free Hessian terms are accumulated only on the
    cloth particles; the selected rigid shapes act as prescribed colliders.

    Call :meth:`update_colliders` before each solver step so the contact
    surface follows the current rigid transforms and velocities.
    """

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
        self.collider_positions = wp.clone(self.collider_local_positions)
        self.collider_velocities = wp.zeros_like(self.collider_local_positions)
        self._step_collider_positions = wp.clone(self.collider_positions)
        self._predicted_collider_positions = wp.clone(self.collider_positions)
        self._body_com = model.body_com
        self._zero_body_qd = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self.device)

        cloth_triangles_np = np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape((-1, 3))
        if cloth_triangles_np.shape != (model.tri_count, 3):
            raise ValueError("model triangle topology must have shape [triangle_count, 3]")
        if np.any(cloth_triangles_np < 0) or np.any(cloth_triangles_np >= model.particle_count):
            raise ValueError("model triangle topology contains an invalid particle index")
        self.cloth_triangles = wp.array(cloth_triangles_np, dtype=wp.int32, device=self.device)
        cloth_edges_np = MeshAdjacency(cloth_triangles_np).edge_indices.astype(np.int32)
        self.cloth_edges = wp.array(cloth_edges_np, dtype=wp.int32, device=self.device)
        self.cloth_rest_positions = wp.clone(model.particle_q)
        self._step_positions = wp.clone(model.particle_q)
        self._predicted_positions = wp.clone(model.particle_q)
        self.cloth_triangle_count = len(cloth_triangles_np)
        self.collider_triangle_count = len(triangles_np)
        self.cloth_edge_count = len(cloth_edges_np)
        self.collider_edge_count = len(edges_np)
        self.cloth_triangle_lower_bounds = wp.empty(self.cloth_triangle_count, dtype=wp.vec3, device=self.device)
        self.cloth_triangle_upper_bounds = wp.empty_like(self.cloth_triangle_lower_bounds)
        self.collider_triangle_lower_bounds = wp.empty(self.collider_triangle_count, dtype=wp.vec3, device=self.device)
        self.collider_triangle_upper_bounds = wp.empty_like(self.collider_triangle_lower_bounds)
        self.cloth_edge_lower_bounds = wp.empty(self.cloth_edge_count, dtype=wp.vec3, device=self.device)
        self.cloth_edge_upper_bounds = wp.empty_like(self.cloth_edge_lower_bounds)
        self.collider_edge_lower_bounds = wp.empty(self.collider_edge_count, dtype=wp.vec3, device=self.device)
        self.collider_edge_upper_bounds = wp.empty_like(self.collider_edge_lower_bounds)
        self._update_triangle_bounds(model.particle_q)
        self._update_edge_bounds(model.particle_q)
        self.cloth_triangle_bvh = wp.Bvh(self.cloth_triangle_lower_bounds, self.cloth_triangle_upper_bounds)
        self.collider_triangle_bvh = wp.Bvh(self.collider_triangle_lower_bounds, self.collider_triangle_upper_bounds)
        self.cloth_edge_bvh = wp.Bvh(self.cloth_edge_lower_bounds, self.cloth_edge_upper_bounds)
        self.collider_edge_bvh = wp.Bvh(self.collider_edge_lower_bounds, self.collider_edge_upper_bounds)
        self.cloth_vertex_face_contacts = _KinematicContactBuffer(1, max_contacts, self.particle_count, self.device)
        self.rigid_vertex_face_contacts = _KinematicContactBuffer(3, max_contacts, self.particle_count, self.device)
        self.edge_edge_contacts = _KinematicEdgeEdgeContactBuffer(max_contacts, self.particle_count, self.device)
        self._velocities: wp.array[wp.vec3] | None = None
        self._anchor_positions: wp.array[wp.vec3] | None = None
        if self.friction > 0.0:
            self._anchor_positions = wp.empty_like(model.particle_q)
        self._dt = 0.0
        self._colliders_updated = False

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
        self._colliders_updated = True

    def begin_step(
        self,
        positions: wp.array[wp.vec3],
        velocities: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Cache step-start cloth state for damping and friction.

        Args:
            positions: Step-start cloth positions [m].
            velocities: Step-start cloth velocities [m/s].
            dt: Simulation time step [s].
        """
        self._validate_particle_vectors((positions, "positions"), (velocities, "velocities"))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self._velocities = velocities
        self._dt = float(dt)
        self._step_positions.assign(positions)
        self._step_collider_positions.assign(self.collider_positions)
        wp.launch(
            predict_positions,
            dim=self.particle_count,
            inputs=[positions, velocities, dt],
            outputs=[self._predicted_positions],
            device=self.device,
        )
        wp.launch(
            predict_positions,
            dim=len(self.collider_positions),
            inputs=[self.collider_positions, self.collider_velocities, dt],
            outputs=[self._predicted_collider_positions],
            device=self.device,
        )
        if self._anchor_positions is not None:
            self._anchor_positions.assign(positions)

    def prepare(self, positions: wp.array[wp.vec3]) -> None:
        """Detect and freeze cross-surface VF and EE contacts."""
        self._validate_particle_vectors((positions, "positions"))
        if self._velocities is None or self._dt <= 0.0:
            raise RuntimeError("begin_step() must be called before prepare()")
        if not self._colliders_updated:
            raise RuntimeError("update_colliders() must be called before prepare()")

        self._update_swept_bounds(positions)
        self.cloth_triangle_bvh.refit()
        self.collider_triangle_bvh.refit()
        self.cloth_edge_bvh.refit()
        self.collider_edge_bvh.refit()
        self.cloth_vertex_face_contacts.clear()
        self.rigid_vertex_face_contacts.clear()
        self.edge_edge_contacts.clear()
        wp.launch(
            detect_cloth_vertex_rigid_face,
            dim=self.particle_count,
            inputs=[
                self.collider_triangle_bvh.id,
                self.thickness,
                self.max_contacts,
                positions,
                self._step_positions,
                self._predicted_positions,
                self.collider_positions,
                self._step_collider_positions,
                self._predicted_collider_positions,
                self.collider_velocities,
                self.collider_triangles,
            ],
            outputs=[
                self.cloth_vertex_face_contacts.ids,
                self.cloth_vertex_face_contacts.weights,
                self.cloth_vertex_face_contacts.directions,
                self.cloth_vertex_face_contacts.depths,
                self.cloth_vertex_face_contacts.rigid_velocities,
                self.cloth_vertex_face_contacts.count,
                self.cloth_vertex_face_contacts.overflow_count,
            ],
            device=self.device,
        )
        wp.launch(
            detect_rigid_vertex_cloth_face,
            dim=len(self.collider_positions),
            inputs=[
                self.cloth_triangle_bvh.id,
                self.thickness,
                self.max_contacts,
                positions,
                self._step_positions,
                self._predicted_positions,
                self.cloth_triangles,
                self.collider_positions,
                self._step_collider_positions,
                self._predicted_collider_positions,
                self.collider_velocities,
            ],
            outputs=[
                self.rigid_vertex_face_contacts.ids,
                self.rigid_vertex_face_contacts.weights,
                self.rigid_vertex_face_contacts.directions,
                self.rigid_vertex_face_contacts.depths,
                self.rigid_vertex_face_contacts.rigid_velocities,
                self.rigid_vertex_face_contacts.count,
                self.rigid_vertex_face_contacts.overflow_count,
            ],
            device=self.device,
        )
        wp.launch(
            detect_cloth_edge_rigid_edge,
            dim=self.cloth_edge_count,
            inputs=[
                self.collider_edge_bvh.id,
                self.thickness,
                self.max_contacts,
                positions,
                self.cloth_rest_positions,
                self.cloth_edges,
                self.collider_positions,
                self.collider_velocities,
                self.collider_edges,
            ],
            outputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.weights,
                self.edge_edge_contacts.directions,
                self.edge_edge_contacts.depths,
                self.edge_edge_contacts.rigid_velocities,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.rigid_edge_vectors,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.overflow_count,
            ],
            device=self.device,
        )
        self.edge_edge_contacts.prepare_hessian(positions)

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add cached normal, damping, and friction forces."""
        self._validate_particle_vectors((positions, "positions"), (output, "output"))
        velocities = self._step_velocities()
        for contacts in self._vf_contact_buffers():
            contacts.accumulate_force(self.stiffness, output)
        self.edge_edge_contacts.accumulate_force(self.stiffness, positions, output)
        for contacts in self._contact_buffers():
            if self.normal_damping > 0.0:
                contacts.accumulate_damping_force(self.normal_damping, self._dt, velocities, output)
            if self.friction > 0.0:
                contacts.accumulate_friction_force(
                    self.stiffness,
                    self.friction,
                    self.friction_epsilon,
                    self._dt,
                    positions,
                    self._friction_anchors(),
                    output,
                )

    def hessian_multiply(
        self,
        positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add cached normal, damping, and friction Hessian-vector products."""
        self._validate_particle_vectors(
            (positions, "positions"),
            (vector, "vector"),
            (output, "output"),
        )
        velocities = self._step_velocities()
        for contacts in self._vf_contact_buffers():
            contacts.hessian_multiply(self.stiffness, vector, output)
        self.edge_edge_contacts.hessian_multiply(self.stiffness, positions, vector, output)
        for contacts in self._contact_buffers():
            if self.normal_damping > 0.0:
                contacts.damping_hessian_multiply(
                    self.normal_damping,
                    self._dt,
                    velocities,
                    vector,
                    output,
                )
            if self.friction > 0.0:
                contacts.friction_hessian_multiply(
                    self.stiffness,
                    self.friction,
                    self.friction_epsilon,
                    self._dt,
                    positions,
                    self._friction_anchors(),
                    vector,
                    output,
                )

    def accumulate_diagonal(self, positions: wp.array[wp.vec3], output: wp.array[wp.mat33]) -> None:
        """Add cached normal, damping, and friction diagonal Hessian blocks."""
        self._validate_particle_vectors((positions, "positions"))
        if output.device != self.device or output.dtype != wp.mat33 or len(output) != self.particle_count:
            raise ValueError(f"output must contain {self.particle_count} wp.mat33 values on {self.device}")
        velocities = self._step_velocities()
        for contacts in self._vf_contact_buffers():
            contacts.accumulate_diagonal(self.stiffness, output)
        self.edge_edge_contacts.accumulate_diagonal(self.stiffness, positions, output)
        for contacts in self._contact_buffers():
            if self.normal_damping > 0.0:
                contacts.accumulate_damping_diagonal(self.normal_damping, self._dt, velocities, output)
            if self.friction > 0.0:
                contacts.accumulate_friction_diagonal(
                    self.stiffness,
                    self.friction,
                    self.friction_epsilon,
                    self._dt,
                    positions,
                    self._friction_anchors(),
                    output,
                )

    def _update_triangle_bounds(self, cloth_positions: wp.array[wp.vec3]) -> None:
        wp.launch(
            update_triangle_bounds,
            dim=self.cloth_triangle_count,
            inputs=[cloth_positions, self.cloth_triangles],
            outputs=[self.cloth_triangle_lower_bounds, self.cloth_triangle_upper_bounds],
            device=self.device,
        )
        wp.launch(
            update_triangle_bounds,
            dim=self.collider_triangle_count,
            inputs=[self.collider_positions, self.collider_triangles],
            outputs=[self.collider_triangle_lower_bounds, self.collider_triangle_upper_bounds],
            device=self.device,
        )

    def _update_edge_bounds(self, cloth_positions: wp.array[wp.vec3]) -> None:
        wp.launch(
            update_edge_bounds,
            dim=self.cloth_edge_count,
            inputs=[cloth_positions, self.cloth_edges],
            outputs=[self.cloth_edge_lower_bounds, self.cloth_edge_upper_bounds],
            device=self.device,
        )
        wp.launch(
            update_edge_bounds,
            dim=self.collider_edge_count,
            inputs=[self.collider_positions, self.collider_edges],
            outputs=[self.collider_edge_lower_bounds, self.collider_edge_upper_bounds],
            device=self.device,
        )

    def _update_swept_bounds(self, cloth_positions: wp.array[wp.vec3]) -> None:
        wp.launch(
            update_swept_triangle_bounds,
            dim=self.cloth_triangle_count,
            inputs=[self._step_positions, self._predicted_positions, cloth_positions, self.cloth_triangles],
            outputs=[self.cloth_triangle_lower_bounds, self.cloth_triangle_upper_bounds],
            device=self.device,
        )
        wp.launch(
            update_swept_triangle_bounds,
            dim=self.collider_triangle_count,
            inputs=[
                self._step_collider_positions,
                self._predicted_collider_positions,
                self.collider_positions,
                self.collider_triangles,
            ],
            outputs=[self.collider_triangle_lower_bounds, self.collider_triangle_upper_bounds],
            device=self.device,
        )
        wp.launch(
            update_swept_edge_bounds,
            dim=self.cloth_edge_count,
            inputs=[self._step_positions, self._predicted_positions, cloth_positions, self.cloth_edges],
            outputs=[self.cloth_edge_lower_bounds, self.cloth_edge_upper_bounds],
            device=self.device,
        )
        wp.launch(
            update_swept_edge_bounds,
            dim=self.collider_edge_count,
            inputs=[
                self._step_collider_positions,
                self._predicted_collider_positions,
                self.collider_positions,
                self.collider_edges,
            ],
            outputs=[self.collider_edge_lower_bounds, self.collider_edge_upper_bounds],
            device=self.device,
        )

    def _contact_buffers(
        self,
    ) -> tuple[_KinematicContactBuffer, _KinematicContactBuffer, _KinematicEdgeEdgeContactBuffer]:
        return self.cloth_vertex_face_contacts, self.rigid_vertex_face_contacts, self.edge_edge_contacts

    def _vf_contact_buffers(self) -> tuple[_KinematicContactBuffer, _KinematicContactBuffer]:
        return self.cloth_vertex_face_contacts, self.rigid_vertex_face_contacts

    def _step_velocities(self) -> wp.array[wp.vec3]:
        if self._velocities is None:
            raise RuntimeError("begin_step() must be called before contact evaluation")
        return self._velocities

    def _friction_anchors(self) -> wp.array[wp.vec3]:
        if self._anchor_positions is None:
            raise RuntimeError("friction anchor storage is unavailable")
        return self._anchor_positions

    def _validate_particle_vectors(self, *arrays: tuple[wp.array[wp.vec3], str]) -> None:
        for array, name in arrays:
            if array.device != self.device or array.dtype != wp.vec3 or len(array) != self.particle_count:
                raise ValueError(f"{name} must contain {self.particle_count} wp.vec3 values on {self.device}")
