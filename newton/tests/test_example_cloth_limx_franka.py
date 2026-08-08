# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


def _mesh_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _count_edge_triangle_intersections(
    edge_positions: np.ndarray,
    edges: np.ndarray,
    triangle_positions: np.ndarray,
    triangles: np.ndarray,
) -> int:
    triangle_points = triangle_positions[triangles]
    triangle_lower = triangle_points.min(axis=1)
    triangle_upper = triangle_points.max(axis=1)
    intersection_count = 0
    epsilon = 1.0e-7
    for edge in edges:
        start, end = edge_positions[edge]
        segment_lower = np.minimum(start, end)
        segment_upper = np.maximum(start, end)
        candidates = np.flatnonzero(
            np.all(segment_upper + epsilon >= triangle_lower, axis=1)
            & np.all(segment_lower - epsilon <= triangle_upper, axis=1)
        )
        if len(candidates) == 0:
            continue

        candidate_points = triangle_points[candidates]
        first = candidate_points[:, 1] - candidate_points[:, 0]
        second = candidate_points[:, 2] - candidate_points[:, 0]
        direction = end - start
        cross_direction_second = np.cross(direction, second)
        determinant = np.einsum("ij,ij->i", first, cross_direction_second)
        nonparallel = np.abs(determinant) > epsilon
        inverse_determinant = np.zeros_like(determinant)
        inverse_determinant[nonparallel] = 1.0 / determinant[nonparallel]
        offset = start - candidate_points[:, 0]
        barycentric_u = inverse_determinant * np.einsum("ij,ij->i", offset, cross_direction_second)
        cross_offset_first = np.cross(offset, first)
        barycentric_v = inverse_determinant * np.einsum(
            "j,ij->i",
            direction,
            cross_offset_first,
        )
        segment_parameter = inverse_determinant * np.einsum("ij,ij->i", second, cross_offset_first)
        intersects = (
            nonparallel
            & (barycentric_u > epsilon)
            & (barycentric_v > epsilon)
            & (barycentric_u + barycentric_v < 1.0 - epsilon)
            & (segment_parameter > epsilon)
            & (segment_parameter < 1.0 - epsilon)
        )
        intersection_count += int(np.count_nonzero(intersects))
    return intersection_count


def _count_mesh_intersections(
    first_positions: np.ndarray,
    first_triangles: np.ndarray,
    second_positions: np.ndarray,
    second_triangles: np.ndarray,
) -> int:
    return _count_edge_triangle_intersections(
        first_positions,
        _mesh_edges(first_triangles),
        second_positions,
        second_triangles,
    ) + _count_edge_triangle_intersections(
        second_positions,
        _mesh_edges(second_triangles),
        first_positions,
        first_triangles,
    )


def _count_points_inside_convex_surface(
    points: np.ndarray,
    surface_positions: np.ndarray,
    surface_triangles: np.ndarray,
) -> int:
    triangle_points = surface_positions[surface_triangles]
    normals = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    relative = points[:, None, :] - triangle_points[None, :, 0, :]
    signed_distances = np.einsum("pfj,fj->pf", relative, normals)
    return int(np.count_nonzero(np.all(signed_distances < -1.0e-7, axis=1)))


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    translation = transform[:3]
    quaternion = transform[3:]
    quaternion_vector = quaternion[:3]
    twice_cross = 2.0 * np.cross(quaternion_vector, points)
    return points + quaternion[3] * twice_cross + np.cross(quaternion_vector, twice_cross) + translation


def _shape_world_vertices(model: newton.Model, state: newton.State, shape: int) -> np.ndarray:
    shape_type = int(model.shape_type.numpy()[shape])
    scale = model.shape_scale.numpy()[shape]
    if shape_type == int(newton.GeoType.BOX):
        vertices = np.asarray(list(itertools.product(*[(-extent, extent) for extent in scale])), dtype=np.float32)
    elif shape_type == int(newton.GeoType.MESH):
        vertices = np.asarray(model.shape_source[shape].vertices, dtype=np.float32) * scale
    else:
        raise AssertionError(f"Unexpected gripper shape type {shape_type}")

    shape_vertices = _transform_points(vertices, model.shape_transform.numpy()[shape])
    body = int(model.shape_body.numpy()[shape])
    return _transform_points(shape_vertices, state.body_q.numpy()[body])


class TestClothLimxFranka(unittest.TestCase):
    def test_square_cloth_grid_has_expected_flat_topology(self):
        """Build a flat square grid with alternating triangle diagonals."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        positions, triangles = module._create_square_cloth_grid(
            grid_cells=20,
            width=0.4,
            center=(0.0, -0.5),
            height=0.205,
        )

        self.assertEqual(positions.shape, (441, 3))
        self.assertEqual(triangles.shape, (800, 3))
        self.assertEqual(positions.dtype, np.float32)
        self.assertEqual(triangles.dtype, np.int32)
        np.testing.assert_allclose(positions[:, 2], 0.205, rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(positions[:, :2].min(axis=0), (-0.2, -0.7), rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(positions[:, :2].max(axis=0), (0.2, -0.3), rtol=0.0, atol=1.0e-7)
        np.testing.assert_array_equal(triangles[:2], ((0, 1, 22), (0, 22, 21)))


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxFrankaCuda(unittest.TestCase):
    def test_scene_separates_table_and_gripper_contact_materials(self):
        """Assign low table friction without weakening gripper friction."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=False))

        self.assertTrue(hasattr(example, "table_shape_indices"))
        self.assertTrue(hasattr(example, "gripper_shape_indices"))
        self.assertTrue(hasattr(example, "table_contact"))
        self.assertTrue(hasattr(example, "gripper_contact"))
        np.testing.assert_array_equal(example.table_shape_indices, example.collider_shape_indices[:1])
        np.testing.assert_array_equal(example.gripper_shape_indices, example.collider_shape_indices[1:])
        self.assertAlmostEqual(example.table_contact.friction, 0.05)
        self.assertAlmostEqual(example.gripper_contact.friction, 0.4)
        self.assertFalse(example.table_contact.enable_ccd)
        self.assertTrue(example.gripper_contact.enable_ccd)
        self.assertEqual(len(example.table_contact.collider_positions), 8)
        self.assertEqual(len(example.gripper_contact.collider_positions), 16)

    def test_scene_hides_robot_collision_shapes_by_default(self):
        """Hide robot collision geometry while retaining it for cloth contact."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=False))

        flags = example.model.shape_flags.numpy()
        robot_shapes = example.model.shape_body.numpy() >= 0
        colliders = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        visible = (flags & int(newton.ShapeFlags.VISIBLE)) != 0
        self.assertTrue(np.any(robot_shapes & colliders))
        self.assertFalse(np.any(robot_shapes & colliders & visible))

    def test_edge_grasp_keeps_complete_gripper_outside_table(self):
        """Keep every gripper mesh and box vertex outside the table volume."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=False))

        table_lower = np.asarray(module.TABLE_CENTER) - np.asarray(module.TABLE_HALF_EXTENTS)
        table_upper = np.asarray(module.TABLE_CENTER) + np.asarray(module.TABLE_HALF_EXTENTS)
        inside_count = 0
        for shape, body in enumerate(example.model.shape_body.numpy()):
            if int(body) < 0:
                continue
            body_label = example.model.body_label[int(body)]
            if not body_label.endswith(("fr3_hand", "fr3_leftfinger", "fr3_rightfinger")):
                continue
            vertices = _shape_world_vertices(example.model, example.state_0, shape)
            inside = np.all((vertices > table_lower + 1.0e-6) & (vertices < table_upper - 1.0e-6), axis=1)
            inside_count += int(np.count_nonzero(inside))

        self.assertEqual(inside_count, 0)

    def test_scene_starts_with_active_square_cloth_above_table(self):
        """Start active square cloth with a graspable overhang at the table edge."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=False))
            flags = example.model.particle_flags.numpy()
            positions = example.state_0.particle_q.numpy()
            collider_positions = example.gripper_contact.collider_positions.numpy()

        self.assertEqual(module.CLOTH_GRID_CELLS, 50)
        self.assertEqual(example.model.particle_count, 2601)
        self.assertGreater(example.model.body_count, 0)
        self.assertTrue(example.model.body_label[example.hand_body].endswith("fr3_hand"))
        self.assertEqual(len(example.collider_shape_indices), 3)
        self.assertTrue(
            np.all(example.model.shape_type.numpy()[example.collider_shape_indices] == int(newton.GeoType.BOX))
        )
        self.assertTrue(np.all((flags & int(newton.ParticleFlags.ACTIVE)) != 0))
        np.testing.assert_allclose(positions, example.cloth_rest_positions, rtol=0.0, atol=1.0e-7)
        self.assertGreater(float(positions[:, 2].min()), example.table_top_z)
        self.assertAlmostEqual(float(positions[:, 1].min()), module.TABLE_FRONT_Y - module.CLOTH_OVERHANG)
        self.assertGreater(float(positions[:, 1].max()), module.TABLE_FRONT_Y + 0.30)
        self.assertLess(float(example.grasp_position[1]), module.TABLE_FRONT_Y)
        finger_centers = np.asarray([collider_positions[0:8].mean(axis=0), collider_positions[8:16].mean(axis=0)])
        finger_separation = np.abs(finger_centers[1] - finger_centers[0])
        self.assertGreater(float(finger_separation[2]), 0.08)
        self.assertLess(float(finger_separation[1]), 0.01)
        self.assertLess(float(collider_positions[:, 1].max()), module.TABLE_FRONT_Y)

    def test_first_step_starts_from_approach_pose_without_striking_cloth(self):
        """Start the robot at its approach pose without sweeping through cloth."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=True))
            initial_centroid = float(example.state_0.particle_q.numpy()[:, 2].mean())

            example.step()

            first_step_centroid = float(example.state_0.particle_q.numpy()[:, 2].mean())

        self.assertLess(abs(first_step_centroid - initial_centroid), 0.002)

    def test_refined_cloth_settles_on_table_before_grasp(self):
        """Settle refined cloth on the table using positional contact alone."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=251), SimpleNamespace(graph_capture=True))
            for _ in range(250):
                wp.capture_launch(example.graph)
            positions = example.state_0.particle_q.numpy()
            velocities = example.state_0.particle_qd.numpy()

        supported = (
            (positions[:, 0] >= module.TABLE_CENTER[0] - module.TABLE_HALF_EXTENTS[0])
            & (positions[:, 0] <= module.TABLE_CENTER[0] + module.TABLE_HALF_EXTENTS[0])
            & (positions[:, 1] >= module.TABLE_FRONT_Y)
            & (positions[:, 1] <= module.TABLE_CENTER[1] + module.TABLE_HALF_EXTENTS[1])
        )
        supported_normal_rms = float(np.sqrt(np.mean(velocities[supported, 2] ** 2)))

        self.assertLess(supported_normal_rms, 0.005)

    def test_grasp_sequence_lifts_and_releases_active_cloth_with_box_contacts(self):
        """Lift and release active cloth using kinematic Franka box contacts."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            frame_count = int(np.ceil(module.SEQUENCE_DURATION * module.FPS)) + 1
            release_frame_count = int(1.6 * module.FPS)
            example = module.Example(
                ViewerNull(num_frames=frame_count + release_frame_count),
                SimpleNamespace(graph_capture=True),
            )
            initial_centroid_height = float(example.state_0.particle_q.numpy()[:, 2].mean())
            maximum_lift = 0.0
            consecutive_raised_frames = 0
            maximum_raised_frames = 0
            preclose_intersection_count = None
            hold_intersection_count = None
            hold_inside_vertex_count = None
            for _ in range(frame_count + release_frame_count):
                example.step()
                example.test_post_step()
                centroid_height = float(example.state_0.particle_q.numpy()[:, 2].mean())
                maximum_lift = max(maximum_lift, centroid_height - initial_centroid_height)
                if centroid_height - initial_centroid_height > 0.10:
                    consecutive_raised_frames += 1
                    maximum_raised_frames = max(maximum_raised_frames, consecutive_raised_frames)
                else:
                    consecutive_raised_frames = 0
                if preclose_intersection_count is None and example.sim_time >= example.preclose_time - 0.01:
                    collider_positions = example.gripper_contact.collider_positions.numpy()
                    collider_triangles = example.gripper_contact.collider_triangles.numpy()
                    preclose_intersection_count = 0
                    for shape_slot in range(len(example.gripper_shape_indices)):
                        vertex_start = 8 * shape_slot
                        triangle_start = 12 * shape_slot
                        preclose_intersection_count += _count_mesh_intersections(
                            example.state_0.particle_q.numpy(),
                            example.model.tri_indices.numpy(),
                            collider_positions[vertex_start : vertex_start + 8],
                            collider_triangles[triangle_start : triangle_start + 12] - vertex_start,
                        )
                if hold_intersection_count is None and example.sim_time >= example.lift_time - 0.01:
                    collider_positions = example.gripper_contact.collider_positions.numpy()
                    collider_triangles = example.gripper_contact.collider_triangles.numpy()
                    hold_intersection_count = _count_mesh_intersections(
                        example.state_0.particle_q.numpy(),
                        example.model.tri_indices.numpy(),
                        collider_positions,
                        collider_triangles,
                    )
                    hold_inside_vertex_count = 0
                    for shape_slot in range(len(example.gripper_shape_indices)):
                        vertex_start = 8 * shape_slot
                        triangle_start = 12 * shape_slot
                        hold_inside_vertex_count += _count_points_inside_convex_surface(
                            example.state_0.particle_q.numpy(),
                            collider_positions[vertex_start : vertex_start + 8],
                            collider_triangles[triangle_start : triangle_start + 12] - vertex_start,
                        )
            example.test_final()
            final_particle_positions = example.state_0.particle_q.numpy()
            finger_bottom = float(example.gripper_contact.collider_positions.numpy()[:, 2].min())

        self.assertEqual(example.sim_substeps, 1)
        self.assertAlmostEqual(example.sim_dt, 0.01)
        self.assertGreater(maximum_lift, 0.10)
        self.assertGreaterEqual(maximum_raised_frames / module.FPS, 0.5)
        self.assertGreater(example.maximum_ccd_binding_count, 0)
        self.assertEqual(preclose_intersection_count, 0)
        self.assertEqual(hold_inside_vertex_count, 0)
        self.assertLessEqual(hold_intersection_count, 20)
        self.assertLess(example.minimum_grasp_error, 0.03)
        self.assertGreater(example.maximum_tcp_height, example.grasp_position[2] + 0.10)
        self.assertLess(float(final_particle_positions[:, 2].max()), finger_bottom - 0.05)


if __name__ == "__main__":
    unittest.main()
