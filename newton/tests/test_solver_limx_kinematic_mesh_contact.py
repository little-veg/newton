# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.limx.constraints.kinematic_mesh_contact import (
    _KinematicContactBuffer,
    _KinematicEdgeEdgeContactBuffer,
)


class TestConstraintKinematicMeshSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = wp.get_device("cuda:0")

    def _make_surface_fixture(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(
            xform=wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity()),
        )
        mesh = newton.Mesh(
            vertices=np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (0.2, 0.0, 0.0),
                    (0.0, 0.3, 0.0),
                ],
                dtype=np.float32,
            ),
            indices=np.asarray((0, 1, 2), dtype=np.int32),
            compute_inertia=False,
        )
        mesh_shape = builder.add_shape_mesh(
            body=body,
            mesh=mesh,
            xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        )
        box_shape = builder.add_shape_box(
            body=body,
            xform=wp.transform(wp.vec3(0.0, 0.5, 0.0), wp.quat_identity()),
            hx=0.1,
            hy=0.2,
            hz=0.3,
        )
        cloth_positions = [
            wp.vec3(-0.5, -0.5, 0.5),
            wp.vec3(0.5, -0.5, 0.5),
            wp.vec3(0.0, 0.5, 0.5),
        ]
        builder.add_particles(
            pos=cloth_positions,
            vel=[wp.vec3(0.0)] * 3,
            mass=[1.0] * 3,
            radius=[0.003] * 3,
        )
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=self.device)
        state = model.state()
        constraint = newton.solvers.ConstraintKinematicMeshContact(
            model=model,
            shape_indices=[mesh_shape, box_shape],
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.4,
            friction_epsilon=1.0e-2,
            max_contacts=64,
        )
        return constraint, state, body

    def test_extracts_mesh_and_box_surface_topology(self):
        """Extract selected mesh and box shapes into one triangle surface."""
        constraint, _state, body = self._make_surface_fixture()

        self.assertEqual(constraint.collider_positions.shape[0], 11)
        self.assertEqual(constraint.collider_triangles.shape, (13, 3))
        self.assertEqual(constraint.collider_edges.shape, (21, 4))
        np.testing.assert_array_equal(constraint.collider_body.numpy(), np.full(11, body, dtype=np.int32))

    def test_update_colliders_transforms_mesh_and_box_vertices(self):
        """Transform selected mesh and box vertices into world space."""
        constraint, state, _body = self._make_surface_fixture()

        constraint.update_colliders(state.body_q, state.body_qd)
        positions = constraint.collider_positions.numpy()
        velocities = constraint.collider_velocities.numpy()

        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(velocities).all())
        np.testing.assert_allclose(positions[0], (1.25, 2.0, 3.0), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(velocities, 0.0, rtol=0.0, atol=0.0)


class TestKinematicContactBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = wp.get_device("cuda:0")

    def test_frozen_contact_force_hvp_and_diagonal_match_dense_reference(self):
        """Match dynamic-only contact operations with a dense frozen Hessian."""
        weights = np.asarray((0.2, 0.3, 0.5), dtype=np.float32)
        direction = np.asarray((0.0, 0.6, 0.8), dtype=np.float32)
        ids = np.asarray((0, 2, 3), dtype=np.int32)
        depth = 0.04
        stiffness = 7.0
        vector_np = np.asarray(
            (
                (0.1, -0.2, 0.3),
                (-0.4, 0.5, -0.6),
                (0.7, -0.8, 0.9),
                (-1.0, 1.1, -1.2),
            ),
            dtype=np.float32,
        )

        contacts = _KinematicContactBuffer(arity=3, capacity=1, particle_count=4, device=self.device)
        contacts.ids.assign(ids.reshape(1, 3))
        contacts.weights.assign(weights.reshape(1, 3))
        contacts.directions.assign(direction.reshape(1, 3))
        contacts.depths.assign(np.asarray((depth,), dtype=np.float32))
        contacts.count.assign(np.asarray((1,), dtype=np.int32))
        force = wp.zeros(4, dtype=wp.vec3, device=self.device)
        product = wp.zeros_like(force)
        diagonal = wp.zeros(4, dtype=wp.mat33, device=self.device)
        vector = wp.array(vector_np, dtype=wp.vec3, device=self.device)

        contacts.accumulate_force(stiffness, force)
        contacts.hessian_multiply(stiffness, vector, product)
        contacts.accumulate_diagonal(stiffness, diagonal)

        dense = np.zeros((12, 12), dtype=np.float64)
        rank_one = stiffness * np.outer(direction, direction)
        expected_force = np.zeros((4, 3), dtype=np.float64)
        expected_diagonal = np.zeros((4, 3, 3), dtype=np.float64)
        for local_i, particle_i in enumerate(ids):
            expected_force[particle_i] += stiffness * depth * weights[local_i] * direction
            expected_diagonal[particle_i] += weights[local_i] ** 2 * rank_one
            for local_j, particle_j in enumerate(ids):
                dense[3 * particle_i : 3 * particle_i + 3, 3 * particle_j : 3 * particle_j + 3] += (
                    weights[local_i] * weights[local_j] * rank_one
                )
        expected_product = (dense @ vector_np.reshape(-1)).reshape((-1, 3))

        np.testing.assert_allclose(force.numpy(), expected_force, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(product.numpy(), expected_product, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(diagonal.numpy(), expected_diagonal, rtol=2.0e-5, atol=2.0e-6)
        self.assertGreaterEqual(float(vector_np.reshape(-1) @ dense @ vector_np.reshape(-1)), 0.0)

    def test_approaching_damping_adds_force_hvp_and_diagonal(self):
        """Add damping force and Hessian only for approaching relative motion."""
        contacts = _KinematicContactBuffer(arity=1, capacity=1, particle_count=1, device=self.device)
        contacts.ids.assign(np.asarray([[0]], dtype=np.int32))
        contacts.weights.assign(np.asarray([[1.0]], dtype=np.float32))
        contacts.directions.assign(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
        contacts.rigid_velocities.assign(np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32))
        contacts.count.assign(np.asarray([1], dtype=np.int32))
        velocities = wp.array([wp.vec3(-2.0, 0.0, 0.0)], dtype=wp.vec3, device=self.device)
        vector = wp.array([wp.vec3(2.0, 0.0, 0.0)], dtype=wp.vec3, device=self.device)
        force = wp.zeros(1, dtype=wp.vec3, device=self.device)
        product = wp.zeros_like(force)
        diagonal = wp.zeros(1, dtype=wp.mat33, device=self.device)

        contacts.accumulate_damping_force(0.5, 1.0 / 60.0, velocities, force)
        contacts.damping_hessian_multiply(0.5, 1.0 / 60.0, velocities, vector, product)
        contacts.accumulate_damping_diagonal(0.5, 1.0 / 60.0, velocities, diagonal)

        np.testing.assert_allclose(force.numpy(), [[0.75, 0.0, 0.0]], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(product.numpy(), [[60.0, 0.0, 0.0]], rtol=0.0, atol=1.0e-5)
        expected_diagonal = np.zeros((1, 3, 3), dtype=np.float32)
        expected_diagonal[0, 0, 0] = 30.0
        np.testing.assert_allclose(diagonal.numpy(), expected_diagonal, rtol=0.0, atol=1.0e-5)

    def test_friction_uses_relative_rigid_displacement_in_all_operator_paths(self):
        """Use rigid relative motion in friction force, HVP, and diagonal paths."""
        contacts = _KinematicContactBuffer(arity=2, capacity=1, particle_count=2, device=self.device)
        contacts.ids.assign(np.asarray([[0, 1]], dtype=np.int32))
        contacts.weights.assign(np.asarray([[0.5, 0.5]], dtype=np.float32))
        contacts.directions.assign(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))
        contacts.depths.assign(np.asarray([0.1], dtype=np.float32))
        contacts.rigid_velocities.assign(np.asarray([[-0.1, 0.0, 0.0]], dtype=np.float32))
        contacts.count.assign(np.asarray([1], dtype=np.int32))
        anchors = wp.zeros(2, dtype=wp.vec3, device=self.device)
        positions = wp.array([wp.vec3(0.02, 0.0, 0.0), wp.vec3(0.02, 0.0, 0.0)], dtype=wp.vec3, device=self.device)
        vector = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=self.device)
        force = wp.zeros(2, dtype=wp.vec3, device=self.device)
        product = wp.zeros_like(force)
        diagonal = wp.zeros(2, dtype=wp.mat33, device=self.device)

        contacts.accumulate_friction_force(10.0, 0.4, 0.2, 0.1, positions, anchors, force)
        contacts.friction_hessian_multiply(10.0, 0.4, 0.2, 0.1, positions, anchors, vector, product)
        contacts.accumulate_friction_diagonal(10.0, 0.4, 0.2, 0.1, positions, anchors, diagonal)

        np.testing.assert_allclose(force.numpy(), [[-0.15, 0.0, 0.0], [-0.15, 0.0, 0.0]], atol=1.0e-6)
        np.testing.assert_allclose(product.numpy(), [[15.0, 0.0, 0.0], [15.0, 0.0, 0.0]], atol=1.0e-5)
        expected_diagonal = np.zeros((2, 3, 3), dtype=np.float32)
        expected_diagonal[:, 0, 0] = 7.5
        expected_diagonal[:, 1, 1] = 7.5
        np.testing.assert_allclose(diagonal.numpy(), expected_diagonal, atol=1.0e-5)


class TestKinematicEdgeEdgeContactBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = wp.get_device("cuda:0")

    def test_mollified_force_hessian_and_diagonal_match_energy(self):
        """Match near-parallel EE force and Hessian paths to the mollified energy."""
        positions_np = np.asarray(((-0.5, 0.0, 0.002), (0.5, 0.0, 0.002)), dtype=np.float32)
        rigid_edge = np.asarray((1.0, 0.01, 0.0), dtype=np.float32)
        direction = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        weights = np.asarray((0.5, 0.5), dtype=np.float32)
        depth = 0.001
        threshold = 1.0e-3 * float(np.dot(rigid_edge, rigid_edge))
        stiffness = 37.0
        vector_np = np.asarray(((0.2, -0.1, 0.3), (-0.4, 0.5, -0.2)), dtype=np.float32)

        contacts = _KinematicEdgeEdgeContactBuffer(capacity=1, particle_count=2, device=self.device)
        contacts.ids.assign(np.asarray([[0, 1]], dtype=np.int32))
        contacts.weights.assign(weights.reshape(1, 2))
        contacts.directions.assign(direction.reshape(1, 3))
        contacts.depths.assign(np.asarray([depth], dtype=np.float32))
        contacts.rigid_edge_vectors.assign(rigid_edge.reshape(1, 3))
        contacts.mollifier_thresholds.assign(np.asarray([threshold], dtype=np.float32))
        contacts.count.assign(np.asarray([1], dtype=np.int32))
        positions = wp.array(positions_np, dtype=wp.vec3, device=self.device)
        vector = wp.array(vector_np, dtype=wp.vec3, device=self.device)
        force = wp.zeros(2, dtype=wp.vec3, device=self.device)
        product = wp.zeros_like(force)
        diagonal = wp.zeros(2, dtype=wp.mat33, device=self.device)

        contacts.prepare_hessian(positions)
        contacts.accumulate_force(stiffness, positions, force)
        contacts.hessian_multiply(stiffness, positions, vector, product)
        contacts.accumulate_diagonal(stiffness, positions, diagonal)

        self.assertEqual(int(contacts.mollifier_active.numpy()[0]), 1)

        def energy(candidate: np.ndarray) -> float:
            displacement = candidate - positions_np
            candidate_depth = depth - float(np.dot(direction, weights @ displacement))
            cross_product = np.cross(candidate[1] - candidate[0], rigid_edge)
            cross_squared = float(np.dot(cross_product, cross_product))
            mollifier = cross_squared * (2.0 * threshold - cross_squared) / (threshold * threshold)
            return 0.5 * stiffness * candidate_depth * candidate_depth * mollifier

        def residual(candidate: np.ndarray) -> np.ndarray:
            displacement = candidate - positions_np
            candidate_depth = depth - float(np.dot(direction, weights @ displacement))
            cross_product = np.cross(candidate[1] - candidate[0], rigid_edge)
            cross_squared = float(np.dot(cross_product, cross_product))
            scale = np.sqrt(max(2.0 * threshold - cross_squared, threshold)) / threshold
            return candidate_depth * scale * cross_product

        epsilon = 1.0e-5
        expected_force = np.zeros((2, 3), dtype=np.float64)
        for particle in range(2):
            for axis in range(3):
                offset = np.zeros((2, 3), dtype=np.float32)
                offset[particle, axis] = epsilon
                expected_force[particle, axis] = -(energy(positions_np + offset) - energy(positions_np - offset)) / (
                    2.0 * epsilon
                )

        jacobian = np.zeros((3, 6), dtype=np.float64)
        for column in range(6):
            offset = np.zeros((2, 3), dtype=np.float32)
            offset.reshape(-1)[column] = epsilon
            jacobian[:, column] = (residual(positions_np + offset) - residual(positions_np - offset)) / (2.0 * epsilon)
        dense = stiffness * jacobian.T @ jacobian
        expected_product = (dense @ vector_np.reshape(-1)).reshape((2, 3))

        np.testing.assert_allclose(force.numpy(), expected_force, rtol=2.0e-3, atol=2.0e-5)
        np.testing.assert_allclose(product.numpy(), expected_product, rtol=2.0e-3, atol=2.0e-4)
        np.testing.assert_allclose(diagonal.numpy()[0], dense[:3, :3], rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(diagonal.numpy()[1], dense[3:, 3:], rtol=2.0e-5, atol=2.0e-5)
        self.assertGreaterEqual(float(vector_np.reshape(-1) @ dense @ vector_np.reshape(-1)), -1.0e-6)

    def test_mollifier_scales_damping_and_friction_loads(self):
        """Scale EE damping and friction consistently with the near-parallel mollifier."""
        positions_np = np.asarray(((-0.5, 0.0, 0.002), (0.5, 0.0, 0.002)), dtype=np.float32)
        rigid_edge = np.asarray((1.0, 0.01, 0.0), dtype=np.float32)
        threshold = 1.0e-3 * float(np.dot(rigid_edge, rigid_edge))
        cross_product = np.cross(positions_np[1] - positions_np[0], rigid_edge)
        cross_squared = float(np.dot(cross_product, cross_product))
        load_scale = cross_squared * (2.0 * threshold - cross_squared) / (threshold * threshold)
        contacts = _KinematicEdgeEdgeContactBuffer(capacity=1, particle_count=2, device=self.device)
        contacts.ids.assign(np.asarray([[0, 1]], dtype=np.int32))
        contacts.weights.assign(np.asarray([[0.5, 0.5]], dtype=np.float32))
        contacts.directions.assign(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))
        contacts.depths.assign(np.asarray([0.001], dtype=np.float32))
        contacts.rigid_edge_vectors.assign(rigid_edge.reshape(1, 3))
        contacts.mollifier_thresholds.assign(np.asarray([threshold], dtype=np.float32))
        contacts.count.assign(np.asarray([1], dtype=np.int32))
        positions = wp.array(positions_np, dtype=wp.vec3, device=self.device)
        contacts.prepare_hessian(positions)

        velocities = wp.array([wp.vec3(0.0, 0.0, -1.0)] * 2, dtype=wp.vec3, device=self.device)
        normal_vector = wp.array([wp.vec3(0.0, 0.0, 1.0)] * 2, dtype=wp.vec3, device=self.device)
        damping_force = wp.zeros(2, dtype=wp.vec3, device=self.device)
        damping_product = wp.zeros_like(damping_force)
        damping_diagonal = wp.zeros(2, dtype=wp.mat33, device=self.device)
        contacts.accumulate_damping_force(0.5, 0.1, velocities, damping_force)
        contacts.damping_hessian_multiply(0.5, 0.1, velocities, normal_vector, damping_product)
        contacts.accumulate_damping_diagonal(0.5, 0.1, velocities, damping_diagonal)

        expected_damping_force = np.zeros((2, 3), dtype=np.float32)
        expected_damping_force[:, 2] = 0.25 * load_scale
        expected_damping_product = np.zeros((2, 3), dtype=np.float32)
        expected_damping_product[:, 2] = 2.5 * load_scale
        expected_damping_diagonal = np.zeros((2, 3, 3), dtype=np.float32)
        expected_damping_diagonal[:, 2, 2] = 1.25 * load_scale
        np.testing.assert_allclose(damping_force.numpy(), expected_damping_force, atol=1.0e-6)
        np.testing.assert_allclose(damping_product.numpy(), expected_damping_product, atol=1.0e-6)
        np.testing.assert_allclose(damping_diagonal.numpy(), expected_damping_diagonal, atol=1.0e-6)

        anchors = positions
        displaced = wp.array(positions_np + np.asarray((0.02, 0.0, 0.0)), dtype=wp.vec3, device=self.device)
        tangent_vector = wp.array([wp.vec3(1.0, 0.0, 0.0)] * 2, dtype=wp.vec3, device=self.device)
        friction_force = wp.zeros(2, dtype=wp.vec3, device=self.device)
        friction_product = wp.zeros_like(friction_force)
        friction_diagonal = wp.zeros(2, dtype=wp.mat33, device=self.device)
        contacts.accumulate_friction_force(10.0, 0.4, 0.2, 0.1, displaced, anchors, friction_force)
        contacts.friction_hessian_multiply(
            10.0,
            0.4,
            0.2,
            0.1,
            displaced,
            anchors,
            tangent_vector,
            friction_product,
        )
        contacts.accumulate_friction_diagonal(10.0, 0.4, 0.2, 0.1, displaced, anchors, friction_diagonal)

        expected_friction_force = np.zeros((2, 3), dtype=np.float32)
        expected_friction_force[:, 0] = -0.002 * load_scale
        expected_friction_product = np.zeros((2, 3), dtype=np.float32)
        expected_friction_product[:, 0] = 0.1 * load_scale
        expected_friction_diagonal = np.zeros((2, 3, 3), dtype=np.float32)
        expected_friction_diagonal[:, 0, 0] = 0.05 * load_scale
        expected_friction_diagonal[:, 1, 1] = 0.05 * load_scale
        np.testing.assert_allclose(friction_force.numpy(), expected_friction_force, atol=1.0e-6)
        np.testing.assert_allclose(friction_product.numpy(), expected_friction_product, atol=1.0e-6)
        np.testing.assert_allclose(friction_diagonal.numpy(), expected_friction_diagonal, atol=1.0e-6)

    def test_inactive_predictive_contact_has_zero_normal_operator(self):
        """Keep inactive predictive EE normal force and Hessian paths zero."""
        positions_np = np.asarray(((-0.5, 0.0, 0.01), (0.5, 0.0, 0.01)), dtype=np.float32)
        contacts = _KinematicEdgeEdgeContactBuffer(capacity=1, particle_count=2, device=self.device)
        contacts.ids.assign(np.asarray([[0, 1]], dtype=np.int32))
        contacts.weights.assign(np.asarray([[0.5, 0.5]], dtype=np.float32))
        contacts.directions.assign(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))
        contacts.depths.assign(np.asarray([-0.01], dtype=np.float32))
        contacts.rigid_edge_vectors.assign(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32))
        contacts.mollifier_thresholds.assign(np.asarray([1.0e-3], dtype=np.float32))
        contacts.count.assign(np.asarray([1], dtype=np.int32))
        positions = wp.array(positions_np, dtype=wp.vec3, device=self.device)
        vector = wp.array([wp.vec3(0.0, 0.0, 1.0)] * 2, dtype=wp.vec3, device=self.device)
        force = wp.zeros(2, dtype=wp.vec3, device=self.device)
        product = wp.zeros_like(force)
        diagonal = wp.zeros(2, dtype=wp.mat33, device=self.device)

        contacts.prepare_hessian(positions)
        contacts.accumulate_force(10.0, positions, force)
        contacts.hessian_multiply(10.0, positions, vector, product)
        contacts.accumulate_diagonal(10.0, positions, diagonal)

        np.testing.assert_allclose(force.numpy(), 0.0, atol=0.0)
        np.testing.assert_allclose(product.numpy(), 0.0, atol=0.0)
        np.testing.assert_allclose(diagonal.numpy(), 0.0, atol=0.0)


class TestConstraintKinematicMeshVF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = wp.get_device("cuda:0")

    def _make_vf_fixture(
        self,
        cloth_positions,
        rigid_vertices,
        cloth_velocities=None,
        dt=1.0 / 600.0,
        prepare=True,
    ):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        rigid_mesh = newton.Mesh(
            vertices=np.asarray(rigid_vertices, dtype=np.float32),
            indices=np.asarray((0, 1, 2), dtype=np.int32),
            compute_inertia=False,
        )
        shape = builder.add_shape_mesh(body=body, mesh=rigid_mesh)
        velocities = cloth_velocities if cloth_velocities is not None else [wp.vec3(0.0)] * 3
        builder.add_particles(
            pos=[wp.vec3(*position) for position in cloth_positions],
            vel=[wp.vec3(*velocity) for velocity in velocities],
            mass=[1.0] * 3,
            radius=[0.003] * 3,
        )
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=self.device)
        state = model.state()
        constraint = newton.solvers.ConstraintKinematicMeshContact(
            model=model,
            shape_indices=[shape],
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.4,
            friction_epsilon=1.0e-2,
            max_contacts=16,
        )
        constraint.update_colliders(state.body_q, state.body_qd)
        constraint.begin_step(state.particle_q, state.particle_qd, dt)
        if prepare:
            constraint.prepare(state.particle_q)
        return constraint, state

    def _make_box_bottom_fixture(self, cloth_z: float):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        positions = ((0.05, 0.05, cloth_z), (0.15, 0.05, cloth_z), (0.1, 0.15, cloth_z))
        builder.add_particles(
            pos=[wp.vec3(*position) for position in positions],
            vel=[wp.vec3(0.0)] * 3,
            mass=[1.0] * 3,
            radius=[0.003] * 3,
        )
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=self.device)
        state = model.state()
        constraint = newton.solvers.ConstraintKinematicMeshContact(
            model=model,
            shape_indices=[shape],
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.0,
            friction_epsilon=1.0e-2,
            max_contacts=64,
        )
        constraint.update_colliders(state.body_q, state.body_qd)
        constraint.begin_step(state.particle_q, state.particle_qd, 0.01)
        return constraint, state

    def _make_box_edge_fixture(self, cloth_z: float):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        positions = ((0.0, -0.15, cloth_z), (0.0, -0.05, cloth_z), (0.001, -0.15, cloth_z))
        builder.add_particles(
            pos=[wp.vec3(*position) for position in positions],
            vel=[wp.vec3(0.0)] * 3,
            mass=[1.0] * 3,
            radius=[0.003] * 3,
        )
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=self.device)
        state = model.state()
        constraint = newton.solvers.ConstraintKinematicMeshContact(
            model=model,
            shape_indices=[shape],
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.0,
            friction_epsilon=1.0e-2,
            max_contacts=64,
        )
        constraint.update_colliders(state.body_q, state.body_qd)
        constraint.begin_step(state.particle_q, state.particle_qd, 0.01)
        return constraint, state

    def _target_edge_contact_count(self, constraint) -> int:
        contacts = constraint.edge_edge_contacts
        count = min(int(contacts.count.numpy()[0]), contacts.capacity)
        ids = np.sort(contacts.ids.numpy()[:count], axis=1)
        return int(np.count_nonzero(np.all(ids == (0, 1), axis=1)))

    def _make_moving_box_ccd_fixture(self, start_x: float = -0.2):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(xform=wp.transform(wp.vec3(start_x, 0.0, 0.0), wp.quat_identity()))
        shape = builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        cloth_positions = np.asarray(((0.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)), dtype=np.float32)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in cloth_positions],
            vel=[wp.vec3(0.0)] * 3,
            mass=[1.0] * 3,
            radius=[0.003] * 3,
        )
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=self.device)
        state = model.state()
        constraint = newton.solvers.ConstraintKinematicMeshContact(
            model=model,
            shape_indices=[shape],
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.0,
            friction_epsilon=1.0e-2,
            max_contacts=64,
            enable_ccd=True,
        )
        constraint.update_colliders(state.body_q, state.body_qd)
        return constraint, state, cloth_positions, model

    def test_ccd_binds_vertex_once_when_box_sweeps_completely_through_it(self):
        """Bind one triangle when a moving box tunnels through a stationary vertex."""
        constraint, state, cloth_positions, _model = self._make_moving_box_ccd_fixture()
        state.body_q.assign([wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())])
        constraint.update_colliders(state.body_q, state.body_qd)
        previous = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        inertia_np = cloth_positions.copy()
        inertia_np[0, 1] += 0.02
        inertia = wp.array(inertia_np, dtype=wp.vec3, device=self.device)
        iterate = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)

        constraint.project_step(previous, inertia, iterate)

        projected = previous.numpy()
        self.assertEqual(int(constraint.ccd_binding_count.numpy()[0]), 1)
        self.assertGreaterEqual(float(projected[0, 0]), 0.303 - 1.0e-5)
        np.testing.assert_allclose(projected[0, 1:], (0.0, 0.0), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(projected[1:], cloth_positions[1:], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(iterate.numpy(), projected, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(inertia.numpy()[0] - projected[0], (0.0, 0.02, 0.0), rtol=0.0, atol=1.0e-6)
        self.assertGreaterEqual(int(constraint.ccd_triangle_ids.numpy()[0]), 0)
        np.testing.assert_array_equal(constraint.ccd_triangle_ids.numpy()[1:], (-1, -1))

        constraint.update_colliders(state.body_q, state.body_qd)
        constraint.project_step(previous, inertia, iterate)

        self.assertEqual(int(constraint.ccd_binding_count.numpy()[0]), 0)
        np.testing.assert_array_equal(constraint.ccd_triangle_ids.numpy(), (-1, -1, -1))

    def test_ccd_skips_stationary_collider(self):
        """Leave cloth unchanged when the collider has no swept motion."""
        constraint, _state, cloth_positions, _model = self._make_moving_box_ccd_fixture()
        previous = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        inertia = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        iterate = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)

        constraint.project_step(previous, inertia, iterate)

        self.assertEqual(int(constraint.ccd_binding_count.numpy()[0]), 0)
        np.testing.assert_allclose(previous.numpy(), cloth_positions, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(constraint.ccd_triangle_ids.numpy(), (-1, -1, -1))

    def test_ccd_does_not_bind_a_touching_surface_moving_away(self):
        """Release a touching vertex when the rigid surface moves away."""
        constraint, state, cloth_positions, _model = self._make_moving_box_ccd_fixture(start_x=-0.103)
        state.body_q.assign([wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity())])
        constraint.update_colliders(state.body_q, state.body_qd)
        previous = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        inertia = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        iterate = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)

        constraint.project_step(previous, inertia, iterate)

        self.assertEqual(int(constraint.ccd_binding_count.numpy()[0]), 0)
        np.testing.assert_allclose(previous.numpy(), cloth_positions, rtol=0.0, atol=0.0)

    def test_ccd_does_not_bind_a_touching_surface_moving_tangentially(self):
        """Leave tangential touching motion to the friction model."""
        constraint, state, cloth_positions, _model = self._make_moving_box_ccd_fixture(start_x=-0.103)
        state.body_q.assign([wp.transform(wp.vec3(-0.103, 0.05, 0.0), wp.quat_identity())])
        constraint.update_colliders(state.body_q, state.body_qd)
        previous = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        inertia = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        iterate = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)

        constraint.project_step(previous, inertia, iterate)

        self.assertEqual(int(constraint.ccd_binding_count.numpy()[0]), 0)
        np.testing.assert_allclose(previous.numpy(), cloth_positions, rtol=0.0, atol=0.0)

    def test_ccd_detects_normal_crossing_with_dominant_tangential_motion(self):
        """Detect a slow normal crossing despite much larger tangential motion."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.005), wp.quat_identity()))
        mesh = newton.Mesh(
            vertices=np.asarray(((-10.0, -10.0, 0.0), (10.0, -10.0, 0.0), (0.0, 10.0, 0.0)), dtype=np.float32),
            indices=np.asarray((0, 1, 2), dtype=np.int32),
            compute_inertia=False,
        )
        shape = builder.add_shape_mesh(body=body, mesh=mesh)
        cloth_positions = np.asarray(((0.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, 20.0)), dtype=np.float32)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in cloth_positions],
            vel=[wp.vec3(0.0)] * 3,
            mass=[1.0] * 3,
            radius=[0.003] * 3,
        )
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=self.device)
        state = model.state()
        constraint = newton.solvers.ConstraintKinematicMeshContact(
            model=model,
            shape_indices=[shape],
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.0,
            friction_epsilon=1.0e-2,
            max_contacts=64,
            enable_ccd=True,
        )
        constraint.update_colliders(state.body_q, state.body_qd)
        state.body_q.assign([wp.transform(wp.vec3(1.0, 0.0, 0.005), wp.quat_identity())])
        constraint.update_colliders(state.body_q, state.body_qd)
        previous = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        inertia = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)
        iterate = wp.array(cloth_positions, dtype=wp.vec3, device=self.device)

        constraint.project_step(previous, inertia, iterate)

        self.assertEqual(int(constraint.ccd_binding_count.numpy()[0]), 1)
        self.assertAlmostEqual(float(constraint.ccd_times.numpy()[0]), 0.2, delta=2.0e-3)
        self.assertGreaterEqual(float(previous.numpy()[0, 2]), 0.008 - 1.0e-5)

    def test_solver_applies_ccd_projection_without_mutating_input_state(self):
        """Apply rigid-sweep CCD before solving cloth while preserving state input."""
        constraint, state_0, cloth_positions, model = self._make_moving_box_ccd_fixture()
        state_1 = model.state()
        state_0.body_q.assign([wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())])
        constraint.update_colliders(state_0.body_q, state_0.body_qd)
        solver = newton.solvers.SolverLIMX(
            model,
            [],
            nonlinear_iterations=1,
            linear_iterations=8,
            dynamic_operator=constraint,
        )

        solver.step(state_0, state_1, model.control(), None, 0.01)

        np.testing.assert_allclose(state_0.particle_q.numpy(), cloth_positions, rtol=0.0, atol=0.0)
        self.assertEqual(int(constraint.ccd_binding_count.numpy()[0]), 1)
        self.assertGreater(float(state_1.particle_q.numpy()[0, 0]), 0.29)

    def test_keeps_cloth_vertex_contact_on_step_start_side_after_crossing(self):
        """Keep a swept cloth vertex on its step-start side of a rigid face."""
        constraint, _state = self._make_vf_fixture(
            cloth_positions=((0.0, 0.0, 0.01), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0)),
            rigid_vertices=((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
            cloth_velocities=((0.0, 0.0, -2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            dt=0.01,
            prepare=False,
        )
        crossed = wp.array(
            ((0.0, 0.0, -0.01), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0)),
            dtype=wp.vec3,
            device=self.device,
        )

        constraint.prepare(crossed)
        contacts = constraint.cloth_vertex_face_contacts

        self.assertGreaterEqual(int(contacts.count.numpy()[0]), 1)
        np.testing.assert_allclose(contacts.directions.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-6)
        self.assertGreater(float(contacts.depths.numpy()[0]), constraint.thickness)

    def test_solver_stops_high_speed_cloth_face_crossing(self):
        """Stop a fast cloth triangle from crossing a rigid face in one step."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body()
        rigid_mesh = newton.Mesh(
            vertices=np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32),
            indices=np.asarray((0, 1, 2), dtype=np.int32),
            compute_inertia=False,
        )
        shape = builder.add_shape_mesh(body=body, mesh=rigid_mesh)
        cloth_positions = np.asarray(((-0.1, -0.1, 0.01), (0.1, -0.1, 0.01), (0.0, 0.1, 0.01)), dtype=np.float32)
        cloth_triangles = np.asarray(((0, 1, 2),), dtype=np.int32)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in cloth_positions],
            vel=[wp.vec3(0.0, 0.0, -2.0)] * 3,
            mass=[0.001] * 3,
            radius=[0.003] * 3,
        )
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=self.device)
        state_0 = model.state()
        state_1 = model.state()
        contact = newton.solvers.ConstraintKinematicMeshContact(
            model=model,
            shape_indices=[shape],
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.0,
            friction_epsilon=1.0e-2,
            max_contacts=64,
        )
        contact.update_colliders(state_0.body_q, state_0.body_qd)
        elasticity = newton.solvers.ConstraintTriangleElastic(
            triangle_indices=cloth_triangles,
            inverse_rest_matrices=model.tri_poses.numpy(),
            rest_areas=model.tri_areas.numpy(),
            stiffnesses=[wp.vec3(1.0)] * len(cloth_triangles),
            particle_count=model.particle_count,
            device=model.device,
        )
        solver = newton.solvers.SolverLIMX(
            model,
            [elasticity],
            nonlinear_iterations=2,
            linear_iterations=50,
            dynamic_operator=contact,
        )

        solver.step(state_0, state_1, model.control(), None, 0.01)

        self.assertGreaterEqual(float(state_1.particle_q.numpy()[:, 2].min()), 0.0)

    def test_keeps_edge_contact_on_step_start_side_after_crossing(self):
        """Keep a swept cloth edge on its step-start side of a rigid edge."""
        constraint, _state = self._make_vf_fixture(
            cloth_positions=((-0.5, 0.0, 0.01), (0.5, 0.0, 0.01), (-0.5, 0.0, 1.0)),
            rigid_vertices=((0.0, -0.5, 0.0), (0.0, 0.5, 0.0), (0.0, -0.5, -1.0)),
            cloth_velocities=((0.0, 0.0, -2.0), (0.0, 0.0, -2.0), (0.0, 0.0, 0.0)),
            dt=0.01,
            prepare=False,
        )
        crossed = wp.array(
            ((-0.5, 0.0, -0.01), (0.5, 0.0, -0.01), (-0.5, 0.0, 1.0)),
            dtype=wp.vec3,
            device=self.device,
        )

        constraint.prepare(crossed)
        contacts = constraint.edge_edge_contacts

        self.assertGreaterEqual(int(contacts.count.numpy()[0]), 1)
        self.assertGreater(float(contacts.depths.numpy()[0]), constraint.thickness)
        np.testing.assert_allclose(contacts.directions.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-6)

    def test_keeps_cloth_on_previous_side_of_moving_rigid_face(self):
        """Keep cloth on its previous side when a rigid face moves across it."""
        constraint, state = self._make_vf_fixture(
            cloth_positions=((0.0, 0.0, 0.01), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0)),
            rigid_vertices=((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
            dt=0.01,
            prepare=False,
        )
        body_q = state.body_q.numpy()
        body_q[0, 2] = 0.02
        state.body_q.assign(body_q)

        constraint.update_colliders(state.body_q, state.body_qd)
        constraint.begin_step(state.particle_q, state.particle_qd, 0.01)
        constraint.prepare(state.particle_q)
        contacts = constraint.cloth_vertex_face_contacts

        self.assertGreaterEqual(int(contacts.count.numpy()[0]), 1)
        np.testing.assert_allclose(contacts.directions.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-6)
        self.assertGreater(float(contacts.depths.numpy()[0]), constraint.thickness)

    def test_pushes_cloth_outward_from_inside_box_face(self):
        """Push a cloth vertex outward when it starts inside a box face."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        cloth_positions = ((0.0, 0.0, 0.099), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0))
        builder.add_particles(
            pos=[wp.vec3(*position) for position in cloth_positions],
            vel=[wp.vec3(0.0)] * 3,
            mass=[1.0] * 3,
            radius=[0.003] * 3,
        )
        builder.add_triangle(0, 1, 2)
        model = builder.finalize(device=self.device)
        state = model.state()
        constraint = newton.solvers.ConstraintKinematicMeshContact(
            model=model,
            shape_indices=[shape],
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.0,
            friction_epsilon=1.0e-2,
            max_contacts=64,
        )
        constraint.update_colliders(state.body_q, state.body_qd)
        constraint.begin_step(state.particle_q, state.particle_qd, 0.01)

        constraint.prepare(state.particle_q)
        contacts = constraint.cloth_vertex_face_contacts
        count = min(int(contacts.count.numpy()[0]), contacts.capacity)
        ids = contacts.ids.numpy()[:count, 0]
        directions = contacts.directions.numpy()[:count]
        depths = contacts.depths.numpy()[:count]
        vertex_contacts = (ids == 0) & (directions[:, 2] > 0.9)

        self.assertTrue(np.any(vertex_contacts))
        self.assertGreater(float(depths[vertex_contacts].max()), constraint.thickness)

    def test_rejects_box_bottom_vertices_from_their_inward_side(self):
        """Reject reverse VF contacts viewed through a box vertex's inward side."""
        constraint, state = self._make_box_bottom_fixture(cloth_z=-0.099)

        constraint.prepare(state.particle_q)

        self.assertEqual(int(constraint.rigid_vertex_face_contacts.count.numpy()[0]), 0)

    def test_keeps_box_bottom_vertices_from_their_outward_side(self):
        """Keep reverse VF contacts in a box vertex's outward normal cone."""
        constraint, state = self._make_box_bottom_fixture(cloth_z=-0.101)

        constraint.prepare(state.particle_q)

        self.assertGreater(int(constraint.rigid_vertex_face_contacts.count.numpy()[0]), 0)

    def test_rejects_box_bottom_edge_from_its_inward_side(self):
        """Reject EE contacts outside a box edge's outward normal cone."""
        constraint, state = self._make_box_edge_fixture(cloth_z=-0.099)

        constraint.prepare(state.particle_q)

        self.assertEqual(self._target_edge_contact_count(constraint), 0)

    def test_keeps_box_bottom_edge_from_its_outward_side(self):
        """Keep EE contacts inside a box edge's outward normal cone."""
        constraint, state = self._make_box_edge_fixture(cloth_z=-0.101)

        constraint.prepare(state.particle_q)

        self.assertGreater(self._target_edge_contact_count(constraint), 0)

    def test_detects_cloth_vertex_against_rigid_face(self):
        """Detect an interior cloth-vertex/rigid-face contact."""
        constraint, _state = self._make_vf_fixture(
            cloth_positions=((0.0, 0.0, 0.002), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0)),
            rigid_vertices=((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        )
        contacts = constraint.cloth_vertex_face_contacts

        self.assertEqual(int(contacts.count.numpy()[0]), 1)
        np.testing.assert_array_equal(contacts.ids.numpy()[0], [0])
        np.testing.assert_allclose(contacts.weights.numpy()[0], [1.0], atol=0.0)
        np.testing.assert_allclose(contacts.directions.numpy()[0], [0.0, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(contacts.depths.numpy()[0], 0.001, atol=1.0e-6)

    def test_detects_rigid_vertex_against_cloth_face(self):
        """Detect a rigid vertex over a cloth-face interior with dynamic face weights."""
        constraint, state = self._make_vf_fixture(
            cloth_positions=((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
            rigid_vertices=((0.0, 0.0, 0.002), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0)),
        )
        contacts = constraint.rigid_vertex_face_contacts

        self.assertEqual(int(contacts.count.numpy()[0]), 1)
        np.testing.assert_array_equal(contacts.ids.numpy()[0], [0, 1, 2])
        np.testing.assert_allclose(contacts.weights.numpy()[0], [-0.25, -0.25, -0.5], atol=1.0e-6)
        np.testing.assert_allclose(contacts.directions.numpy()[0], [0.0, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(contacts.depths.numpy()[0], 0.001, atol=1.0e-6)

        force = wp.zeros(3, dtype=wp.vec3, device=self.device)
        constraint.accumulate_force(state.particle_q, force)
        self.assertLess(float(force.numpy()[:, 2].sum()), 0.0)

    def test_detects_interior_cloth_edge_against_rigid_edge(self):
        """Detect an interior cloth-edge/rigid-edge contact."""
        constraint, _state = self._make_vf_fixture(
            cloth_positions=((-0.5, 0.0, 0.002), (0.5, 0.0, 0.002), (-0.5, 0.0, 1.0)),
            rigid_vertices=((0.0, -0.5, 0.0), (0.0, 0.5, 0.0), (0.0, -0.5, -1.0)),
        )
        contacts = constraint.edge_edge_contacts

        self.assertEqual(int(contacts.count.numpy()[0]), 1)
        np.testing.assert_array_equal(contacts.ids.numpy()[0], [0, 1])
        np.testing.assert_allclose(contacts.weights.numpy()[0], [0.5, 0.5], atol=1.0e-6)
        np.testing.assert_allclose(contacts.directions.numpy()[0], [0.0, 0.0, 1.0], atol=1.0e-6)
        np.testing.assert_allclose(contacts.depths.numpy()[0], 0.001, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
