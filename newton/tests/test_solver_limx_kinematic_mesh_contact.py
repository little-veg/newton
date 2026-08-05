# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.limx.constraints.kinematic_mesh_contact import _KinematicContactBuffer


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
