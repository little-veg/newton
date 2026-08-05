# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a LIMX cloth onto a static palm-up Franka wrist and gripper."""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

FPS = 100
SIM_SUBSTEPS = 1
CLOTH_GRID_CELLS = 20
CLOTH_WIDTH = 0.4
CLOTH_CENTER = (0.0, -0.5)
CLOTH_HEIGHT = 0.52
CLOTH_THICKNESS = 0.003
CLOTH_MASS = 0.2
FRANKA_BASE = (-0.5, -0.5, -0.1)
FRANKA_Q = (
    -2.3049769,
    -0.41451603,
    2.3626099,
    -2.3683839,
    -2.5640166,
    0.54478306,
    1.1920885,
    0.04,
    0.04,
)
COLLIDER_BODY_SUFFIXES = (
    "fr3_link6",
    "fr3_link7",
    "fr3_hand",
    "fr3_leftfinger",
    "fr3_rightfinger",
)


def _create_square_cloth_grid(
    grid_cells: int,
    width: float,
    center: tuple[float, float],
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_side = grid_cells + 1
    positions = np.empty((grid_side * grid_side, 3), dtype=np.float32)
    triangles: list[tuple[int, int, int]] = []

    for y in range(grid_side):
        for x in range(grid_side):
            index = y * grid_side + x
            positions[index] = (
                center[0] - 0.5 * width + width * x / grid_cells,
                center[1] - 0.5 * width + width * y / grid_cells,
                height,
            )

    for y in range(grid_cells):
        for x in range(grid_cells):
            lower_left = y * grid_side + x
            lower_right = lower_left + 1
            upper_left = lower_left + grid_side
            upper_right = upper_left + 1
            if (x + y) % 2 == 0:
                triangles.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
            else:
                triangles.extend(((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left)))

    return positions, np.asarray(triangles, dtype=np.int32)


def _select_collider_shapes(model: newton.Model) -> list[int]:
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape_bodies = model.shape_body.numpy()
    shape_flags = model.shape_flags.numpy()
    selected: list[int] = []
    for shape in range(model.shape_count):
        body = int(shape_bodies[shape])
        if body < 0 or (int(shape_flags[shape]) & collision_flag) == 0:
            continue
        if model.body_label[body].endswith(COLLIDER_BODY_SUFFIXES):
            selected.append(shape)
    if not selected:
        raise ValueError("Franka wrist and gripper collision shapes were not found")
    return selected


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.use_graph = bool(getattr(args, "graph_capture", True))

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(*FRANKA_BASE), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            force_show_colliders=False,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

        positions, triangles = _create_square_cloth_grid(
            grid_cells=CLOTH_GRID_CELLS,
            width=CLOTH_WIDTH,
            center=CLOTH_CENTER,
            height=CLOTH_HEIGHT,
        )
        particle_count = len(positions)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in positions],
            vel=[wp.vec3(0.0)] * particle_count,
            mass=[CLOTH_MASS / particle_count] * particle_count,
            radius=[CLOTH_THICKNESS] * particle_count,
        )
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        builder.color()

        self.model = builder.finalize(requires_grad=False)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.collider_shape_indices = _select_collider_shapes(self.model)
        self.triangle_indices = triangles
        self.initial_positions = positions.copy()

        edge_rows = newton.utils.MeshAdjacency(triangles).edge_indices
        interior_edges = edge_rows[edge_rows[:, 1] >= 0]
        dihedral_indices = interior_edges[:, [2, 3, 0, 1]]
        static_constraints = [
            newton.solvers.ConstraintTriangleElastic(
                triangle_indices=triangles,
                inverse_rest_matrices=self.model.tri_poses.numpy(),
                rest_areas=self.model.tri_areas.numpy(),
                stiffnesses=[wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(triangles),
                particle_count=particle_count,
                device=self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                dihedral_indices=dihedral_indices,
                rest_positions=positions,
                stiffness=1.0e-5,
                particle_count=particle_count,
                device=self.model.device,
            ),
        ]
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=CLOTH_THICKNESS,
            stiffness=None,
            max_contacts=65536,
            stiffness_factors=(0.5, 0.3, 1.5),
            friction=0.4,
            friction_epsilon=1.0e-2,
        )
        self.kinematic_contact = newton.solvers.ConstraintKinematicMeshContact(
            model=self.model,
            shape_indices=self.collider_shape_indices,
            thickness=CLOTH_THICKNESS,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.4,
            friction_epsilon=1.0e-2,
            max_contacts=65536,
        )
        dynamic_constraints = newton.solvers.ConstraintGroupDynamic([self.self_collision, self.kinematic_contact])
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            static_constraints,
            nonlinear_iterations=2,
            linear_iterations=50,
            velocity_damping=0.998,
            dynamic_operator=dynamic_constraints,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.maximum_contact_count = 0
        self.maximum_overflow_count = 0
        self.maximum_contact_depth = 0.0
        self.maximum_attempted_contact_counts = {"cloth_vf": 0, "rigid_vf": 0, "ee": 0}
        self.maximum_retained_contact_counts = {"cloth_vf": 0, "rigid_vf": 0, "ee": 0}
        self.velocity_rms_history: list[float] = []
        self.final_rms_speed = float("inf")
        self.supported = False

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(wp.vec3(0.75, -1.15, 0.75), -16.0, 128.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(0.0, -0.5, 0.34))

        self.use_graph = self.use_graph and self.model.device.is_cuda
        self.capture()

    def capture(self):
        self.graph = None
        if self.use_graph:
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
                self.simulate()
            if capture.graph is None:
                raise RuntimeError(f"Graph capture failed on device {self.model.device}")
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.kinematic_contact.update_colliders(self.state_0.body_q, self.state_0.body_qd)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.assign(self.state_1)

    def _update_metrics(self):
        contact_buffers = (
            ("cloth_vf", self.kinematic_contact.cloth_vertex_face_contacts),
            ("rigid_vf", self.kinematic_contact.rigid_vertex_face_contacts),
            ("ee", self.kinematic_contact.edge_edge_contacts),
        )
        contact_count = 0
        overflow_count = 0
        maximum_depth = 0.0
        for name, contacts in contact_buffers:
            attempted = int(contacts.count.numpy()[0])
            retained = min(attempted, contacts.capacity)
            self.maximum_attempted_contact_counts[name] = max(self.maximum_attempted_contact_counts[name], attempted)
            self.maximum_retained_contact_counts[name] = max(self.maximum_retained_contact_counts[name], retained)
            contact_count += retained
            overflow_count += int(contacts.overflow_count.numpy()[0])
            if retained > 0:
                maximum_depth = max(maximum_depth, float(contacts.depths.numpy()[:retained].max()))
        velocities = self.state_0.particle_qd.numpy()
        rms_speed = float(np.sqrt(np.mean(np.sum(velocities * velocities, axis=1))))
        self.maximum_contact_count = max(self.maximum_contact_count, contact_count)
        self.maximum_overflow_count = max(self.maximum_overflow_count, overflow_count)
        self.maximum_contact_depth = max(self.maximum_contact_depth, maximum_depth)
        self.velocity_rms_history.append(rms_speed)

    def step(self):
        if self.graph is not None:
            with wp.ScopedDevice(self.model.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._update_metrics()

    def test_post_step(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX Franka cloth-drop state contains non-finite values")
        if float(positions[:, 2].min()) < -0.5:
            raise AssertionError("LIMX cloth fell catastrophically through the Franka")
        if self.maximum_overflow_count > 0:
            raise AssertionError(f"LIMX Franka contacts overflowed by {self.maximum_overflow_count}")

    def test_final(self):
        self.test_post_step()
        if self.maximum_contact_count <= 0:
            raise AssertionError("LIMX cloth never contacted the Franka")
        self.final_rms_speed = float(np.mean(self.velocity_rms_history[-min(FPS, len(self.velocity_rms_history)) :]))
        centroid_height = float(self.state_0.particle_q.numpy()[:, 2].mean())
        self.supported = centroid_height > 0.2
        print(
            "LIMX Franka drop diagnostics: "
            f"attempted={self.maximum_attempted_contact_counts}, "
            f"retained={self.maximum_retained_contact_counts}, "
            f"overflow={self.maximum_overflow_count}, "
            f"max_depth={self.maximum_contact_depth:.6f} m, "
            f"final_rms={self.final_rms_speed:.6f} m/s, "
            f"supported={self.supported}"
        )
        if len(self.velocity_rms_history) >= 5 * FPS:
            if self.final_rms_speed >= 0.02:
                raise AssertionError(f"LIMX cloth did not settle: final RMS speed is {self.final_rms_speed:.4f} m/s")
            if not self.supported:
                raise AssertionError(f"LIMX cloth was not supported: centroid height is {centroid_height:.4f} m")

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

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable CUDA graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
