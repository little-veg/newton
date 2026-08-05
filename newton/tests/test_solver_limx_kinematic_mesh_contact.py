# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
