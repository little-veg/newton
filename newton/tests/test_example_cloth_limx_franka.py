# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
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
    def test_scene_starts_with_active_square_cloth_above_table(self):
        """Start every square-cloth particle active above the table."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=False))
            flags = example.model.particle_flags.numpy()
            positions = example.state_0.particle_q.numpy()

        self.assertEqual(example.model.particle_count, 441)
        self.assertGreater(example.model.body_count, 0)
        self.assertTrue(example.model.body_label[example.hand_body].endswith("fr3_hand"))
        self.assertEqual(len(example.collider_shape_indices), 3)
        self.assertTrue(
            np.all(example.model.shape_type.numpy()[example.collider_shape_indices] == int(newton.GeoType.BOX))
        )
        self.assertTrue(np.all((flags & int(newton.ParticleFlags.ACTIVE)) != 0))
        np.testing.assert_allclose(positions, example.cloth_rest_positions, rtol=0.0, atol=1.0e-7)
        self.assertGreater(float(positions[:, 2].min()), example.table_top_z)

    def test_first_step_starts_from_approach_pose_without_striking_cloth(self):
        """Start the robot at its approach pose without sweeping through cloth."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=True))
            initial_centroid = float(example.state_0.particle_q.numpy()[:, 2].mean())

            example.step()

            first_step_centroid = float(example.state_0.particle_q.numpy()[:, 2].mean())

        self.assertLess(abs(first_step_centroid - initial_centroid), 0.002)

    def test_grasp_sequence_lifts_active_cloth_with_box_contacts(self):
        """Lift active cloth using kinematic Franka box contacts and friction."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            frame_count = int(np.ceil(module.SEQUENCE_DURATION * module.FPS)) + 1
            example = module.Example(ViewerNull(num_frames=frame_count), SimpleNamespace(graph_capture=True))
            initial_centroid_height = float(example.state_0.particle_q.numpy()[:, 2].mean())
            maximum_lift = 0.0
            consecutive_raised_frames = 0
            maximum_raised_frames = 0
            hold_intersection_count = None
            hold_inside_vertex_count = None
            for _ in range(frame_count):
                example.step()
                example.test_post_step()
                centroid_height = float(example.state_0.particle_q.numpy()[:, 2].mean())
                maximum_lift = max(maximum_lift, centroid_height - initial_centroid_height)
                if centroid_height - initial_centroid_height > 0.10:
                    consecutive_raised_frames += 1
                    maximum_raised_frames = max(maximum_raised_frames, consecutive_raised_frames)
                else:
                    consecutive_raised_frames = 0
                if hold_intersection_count is None and example.sim_time >= example.key_times[4] - 0.01:
                    collider_positions = example.kinematic_contact.collider_positions.numpy()
                    collider_triangles = example.kinematic_contact.collider_triangles.numpy()
                    hold_intersection_count = _count_mesh_intersections(
                        example.state_0.particle_q.numpy(),
                        example.model.tri_indices.numpy(),
                        collider_positions,
                        collider_triangles,
                    )
                    hold_inside_vertex_count = 0
                    for shape_slot in range(1, len(example.collider_shape_indices)):
                        vertex_start = 8 * shape_slot
                        triangle_start = 12 * shape_slot
                        hold_inside_vertex_count += _count_points_inside_convex_surface(
                            example.state_0.particle_q.numpy(),
                            collider_positions[vertex_start : vertex_start + 8],
                            collider_triangles[triangle_start : triangle_start + 12] - vertex_start,
                        )
            example.test_final()

        self.assertEqual(example.sim_substeps, 1)
        self.assertAlmostEqual(example.sim_dt, 0.01)
        self.assertGreater(maximum_lift, 0.10)
        self.assertGreaterEqual(maximum_raised_frames / module.FPS, 0.5)
        self.assertEqual(hold_inside_vertex_count, 0)
        self.assertLessEqual(hold_intersection_count, 20)
        self.assertLess(example.minimum_grasp_error, 0.03)
        self.assertGreater(example.maximum_tcp_height, example.grasp_position[2] + 0.10)


if __name__ == "__main__":
    unittest.main()
