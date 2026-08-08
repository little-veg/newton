# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fold a full-resolution LIMX cloth in half on a table."""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

FPS = 100
SIM_SUBSTEPS = 1
GRID_CELLS = 100
CLOTH_WIDTH = 1.0
CLOTH_MASS = 0.3
CLOTH_HEIGHT = 0.205
COLLISION_THICKNESS = 0.003
FINAL_LAYER_GAP = 0.006
FOLD_DURATION = 4.0
HOLD_DURATION = 2.0
TABLE_CENTER = (0.0, 0.0, 0.1)
TABLE_HALF_EXTENTS = (0.65, 0.65, 0.1)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_HALF_EXTENTS[2]


def _create_square_cloth_grid(grid_cells: int, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    if grid_cells <= 0:
        raise ValueError("grid_cells must be positive")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and positive")
    if not np.isfinite(height):
        raise ValueError("height must be finite")

    grid_side = grid_cells + 1
    positions = np.empty((grid_side * grid_side, 3), dtype=np.float32)
    for y in range(grid_side):
        for x in range(grid_side):
            index = y * grid_side + x
            positions[index] = (
                -0.5 * width + width * x / grid_cells,
                -0.5 * width + width * y / grid_cells,
                height,
            )

    triangles = np.empty((2 * grid_cells * grid_cells, 3), dtype=np.int32)
    triangle = 0
    for y in range(grid_cells):
        for x in range(grid_cells):
            lower_left = y * grid_side + x
            lower_right = lower_left + 1
            upper_left = lower_left + grid_side
            upper_right = upper_left + 1
            if (x + y) % 2 == 0:
                triangles[triangle] = (lower_left, lower_right, upper_right)
                triangles[triangle + 1] = (lower_left, upper_right, upper_left)
            else:
                triangles[triangle] = (lower_left, lower_right, upper_left)
                triangles[triangle + 1] = (lower_right, upper_right, upper_left)
            triangle += 2
    return positions, triangles


def _compute_fold_boundary_targets(rest_targets: np.ndarray, angle: float, phase: float) -> np.ndarray:
    rest_targets = np.asarray(rest_targets, dtype=np.float32)
    if rest_targets.ndim != 2 or rest_targets.shape[1] != 3 or len(rest_targets) == 0:
        raise ValueError("rest_targets must have shape [target_count, 3]")
    if not np.isfinite(rest_targets).all():
        raise ValueError("rest_targets must be finite")
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    if not np.isfinite(phase) or phase < 0.0 or phase > 1.0:
        raise ValueError("phase must be finite and between zero and one")

    targets = rest_targets.copy()
    radius = rest_targets[:, 0]
    targets[:, 0] = radius * math.cos(angle)
    targets[:, 2] = rest_targets[:, 2] + radius * math.sin(angle) + FINAL_LAYER_GAP * phase
    return targets


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt
        self.sim_time = 0.0
        self.fold_duration = FOLD_DURATION
        self.hold_duration = HOLD_DURATION

        positions, triangles = _create_square_cloth_grid(GRID_CELLS, CLOTH_WIDTH, CLOTH_HEIGHT)
        particle_count = len(positions)
        grid_side = GRID_CELLS + 1

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        self.table_body = builder.add_body(
            xform=wp.transform(wp.vec3(*TABLE_CENTER), wp.quat_identity()),
            label="table",
            is_kinematic=True,
        )
        self.table_shape = builder.add_shape_box(
            body=self.table_body,
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            color=wp.vec3(0.32, 0.34, 0.38),
            label="table",
        )
        builder.add_particles(
            pos=[wp.vec3(*position) for position in positions],
            vel=[wp.vec3(0.0)] * particle_count,
            mass=[CLOTH_MASS / particle_count] * particle_count,
            radius=[COLLISION_THICKNESS] * particle_count,
        )
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        builder.color()

        self.model = builder.finalize(requires_grad=False)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.triangle_indices = triangles
        self.cloth_rest_positions = positions.copy()
        self.left_boundary_indices = [y * grid_side for y in range(grid_side)]
        self.right_boundary_indices = [y * grid_side + GRID_CELLS for y in range(grid_side)]
        self.anchor_indices = self.left_boundary_indices + self.right_boundary_indices
        self.boundary_particle_count = grid_side
        self.anchor_rest_targets = positions[self.anchor_indices].copy()

        inverse_rest_matrices = self.model.tri_poses.numpy()
        edge_rows = newton.utils.MeshAdjacency(triangles).edge_indices
        dihedral_indices = edge_rows[edge_rows[:, 1] >= 0][:, [2, 3, 0, 1]]
        self.anchor_constraint = newton.solvers.ConstraintAnchor(
            self.anchor_indices,
            [wp.vec3(*target) for target in self.anchor_rest_targets],
            [1.0e7] * len(self.anchor_indices),
            particle_count,
            self.model.device,
        )
        static_constraints = [
            self.anchor_constraint,
            newton.solvers.ConstraintTriangleElastic(
                triangles,
                inverse_rest_matrices,
                self.model.tri_areas.numpy(),
                [wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(triangles),
                particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                dihedral_indices,
                positions,
                1.0e-5,
                particle_count,
                self.model.device,
            ),
        ]
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=COLLISION_THICKNESS,
            stiffness=None,
            max_contacts=262144,
            stiffness_factors=(0.5, 0.3, 1.5),
            friction=0.3,
            friction_epsilon=1.0e-2,
        )
        self.table_contact = newton.solvers.ConstraintKinematicMeshContact(
            model=self.model,
            shape_indices=[self.table_shape],
            thickness=COLLISION_THICKNESS,
            stiffness=2.0e4,
            normal_damping=0.0,
            friction=0.05,
            friction_epsilon=1.0e-2,
            max_contacts=65536,
            enable_ccd=False,
        )
        dynamic_constraints = newton.solvers.ConstraintGroupDynamic([self.self_collision, self.table_contact])
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            static_constraints,
            nonlinear_iterations=1,
            linear_iterations=50,
            velocity_damping=1.0,
            dynamic_operator=dynamic_constraints,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.table_contact.update_colliders(self.state_0.body_q, self.state_0.body_qd)

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(wp.vec3(1.55, -1.75, 1.25), -24.0, 138.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.45))
        self.capture()

    def _compute_anchor_targets(self, phase: float) -> np.ndarray:
        smooth_phase = phase * phase * (3.0 - 2.0 * phase)
        targets = self.anchor_rest_targets.copy()
        right_rest = self.anchor_rest_targets[self.boundary_particle_count :]
        targets[self.boundary_particle_count :] = _compute_fold_boundary_targets(
            right_rest,
            math.pi * smooth_phase,
            smooth_phase,
        )
        return targets

    def capture(self):
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.state_0.assign(self.state_1)

    def step(self):
        phase = min(self.sim_time / self.fold_duration, 1.0)
        self.anchor_constraint.targets.assign(self._compute_anchor_targets(phase))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX table-fold scene contains non-finite state")
        right_x = float(positions[self.right_boundary_indices, 0].mean())
        if right_x >= -0.45:
            raise AssertionError(f"Driven cloth boundary reached only x={right_x:.4f} m")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/cloth",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            backface_culling=False,
            color=(0.95, 0.68, 0.05),
            roughness=0.9,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int((FOLD_DURATION + HOLD_DURATION) * FPS))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
