# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grasp active LIMX cloth with kinematic Franka collision boxes."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.viewer import ViewerNull

FPS = 100
SIM_SUBSTEPS = 1
TABLE_CENTER = (0.0, -0.5, 0.1)
TABLE_HALF_EXTENTS = (0.4, 0.4, 0.1)
TABLE_TOP_Z = 0.2
CLOTH_CENTER = (0.0, -0.5)
CLOTH_WIDTH = 0.4
CLOTH_GRASP_Y = CLOTH_CENTER[1]
CLOTH_GRID_CELLS = 20
CLOTH_HEIGHT = TABLE_TOP_Z + 0.005
CLOTH_PARTICLE_RADIUS = 0.003
CLOTH_MASS = 0.2
FRANKA_BASE = (-0.5, -0.5, -0.1)
FRANKA_Q = (
    -3.6802115e-03,
    2.3901723e-02,
    3.6804110e-03,
    -2.3683236,
    -1.2918962e-04,
    2.3922248,
    7.85492e-01,
    0.04,
    0.04,
)
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)
GRIPPER_CLOSED = 0.0029
LIFT_HEIGHT = 0.42
SEQUENCE_DURATION = 6.4
COLLIDER_BODY_SUFFIXES = (
    "fr3_leftfinger",
    "fr3_rightfinger",
)


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


@wp.kernel
def compute_joint_qd(
    target_q: wp.array[float],
    current_q: wp.array[float],
    out_qd: wp.array[float],
    inv_frame_dt: float,
):
    i = wp.tid()
    out_qd[i] = (target_q[i] - current_q[i]) * inv_frame_dt


@wp.kernel
def compute_tcp_position(
    body_q: wp.array[wp.transform],
    body_index: int,
    link_offset: wp.vec3,
    tcp_position: wp.array[wp.vec3],
):
    tcp_position[0] = wp.transform_point(body_q[body_index], link_offset)


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


def _find_label_index(labels: list[str], suffix: str) -> int:
    for index, label in enumerate(labels):
        if label.endswith(suffix):
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")


def _select_collider_shapes(model: newton.Model) -> list[int]:
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape_bodies = model.shape_body.numpy()
    shape_flags = model.shape_flags.numpy()
    shape_types = model.shape_type.numpy()
    shape_transforms = model.shape_transform.numpy()
    table_shape: int | None = None
    finger_shapes: dict[str, int] = {}
    for shape in range(model.shape_count):
        if (int(shape_flags[shape]) & collision_flag) == 0:
            continue
        if int(shape_types[shape]) != int(newton.GeoType.BOX):
            continue
        body = int(shape_bodies[shape])
        if body < 0:
            if model.shape_label[shape] == "table":
                table_shape = shape
            continue
        body_label = model.body_label[body]
        for suffix in COLLIDER_BODY_SUFFIXES:
            if not body_label.endswith(suffix):
                continue
            current = finger_shapes.get(suffix)
            if current is None or shape_transforms[shape, 2] > shape_transforms[current, 2]:
                finger_shapes[suffix] = shape
    if table_shape is None or len(finger_shapes) != len(COLLIDER_BODY_SUFFIXES):
        raise ValueError("Franka finger-pad and table box collision shapes were not found")
    return [table_shape, *(finger_shapes[suffix] for suffix in COLLIDER_BODY_SUFFIXES)]


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.table_top_z = TABLE_TOP_Z
        self.use_graph = bool(getattr(args, "graph_capture", True))
        self.collect_metrics = isinstance(viewer, ViewerNull) or bool(getattr(args, "test", False))

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(*FRANKA_BASE), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*TABLE_CENTER), wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            color=wp.vec3(0.35, 0.37, 0.42),
            label="table",
        )

        positions, triangles = _create_square_cloth_grid(
            grid_cells=CLOTH_GRID_CELLS,
            width=CLOTH_WIDTH,
            center=CLOTH_CENTER,
            height=CLOTH_HEIGHT,
        )
        self.cloth_rest_positions = positions.copy()
        self.triangle_indices = triangles
        particle_count = len(positions)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in positions],
            vel=[wp.vec3(0.0)] * particle_count,
            mass=[CLOTH_MASS / particle_count] * particle_count,
            radius=[CLOTH_PARTICLE_RADIUS] * particle_count,
        )
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        builder.color()

        self.model = builder.finalize(requires_grad=False)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.hand_body = _find_label_index(self.model.body_label, "fr3_hand")
        self.collider_shape_indices = _select_collider_shapes(self.model)
        self.cloth_particle_indices = np.arange(self.model.particle_count, dtype=np.int32)
        self.inverse_rest_matrices = self.model.tri_poses.numpy()
        edge_rows = newton.utils.MeshAdjacency(triangles).edge_indices
        self.dihedral_indices = edge_rows[edge_rows[:, 1] >= 0][:, [2, 3, 0, 1]]

        static_constraints = [
            newton.solvers.ConstraintTriangleElastic(
                triangle_indices=triangles,
                inverse_rest_matrices=self.inverse_rest_matrices,
                rest_areas=self.model.tri_areas.numpy(),
                stiffnesses=[wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(triangles),
                particle_count=particle_count,
                device=self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                dihedral_indices=self.dihedral_indices,
                rest_positions=positions,
                stiffness=1.0e-5,
                particle_count=particle_count,
                device=self.model.device,
            ),
        ]
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=CLOTH_PARTICLE_RADIUS,
            stiffness=None,
            max_contacts=65536,
            stiffness_factors=(0.5, 0.3, 1.5),
            friction=0.4,
            friction_epsilon=1.0e-2,
        )
        self.kinematic_contact = newton.solvers.ConstraintKinematicMeshContact(
            model=self.model,
            shape_indices=self.collider_shape_indices,
            thickness=CLOTH_PARTICLE_RADIUS,
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
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self._build_keyframes()
        self._build_ik()
        self._initialize_robot_pose()

        self.maximum_contact_count = 0
        self.maximum_overflow_count = 0
        self.pre_lift_centroid_height: float | None = None
        self.maximum_cloth_lift = 0.0
        self.captured_hold_duration = 0.0

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(wp.vec3(0.85, -1.45, 0.95), -22.0, 125.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(0.0, -0.5, 0.25))

        self.use_graph = self.use_graph and self.model.device.is_cuda
        self.capture()

    def _build_keyframes(self):
        poses = np.asarray(
            [
                [1.5, 0.0, CLOTH_GRASP_Y, LIFT_HEIGHT, *GRIPPER_DOWN, 0.04],
                [1.2, 0.0, CLOTH_GRASP_Y, CLOTH_HEIGHT, *GRIPPER_DOWN, 0.04],
                [0.8, 0.0, CLOTH_GRASP_Y, CLOTH_HEIGHT, *GRIPPER_DOWN, GRIPPER_CLOSED],
                [1.5, 0.0, CLOTH_GRASP_Y, LIFT_HEIGHT, *GRIPPER_DOWN, GRIPPER_CLOSED],
                [0.6, 0.0, CLOTH_GRASP_Y, LIFT_HEIGHT, *GRIPPER_DOWN, GRIPPER_CLOSED],
                [0.8, 0.0, CLOTH_GRASP_Y, LIFT_HEIGHT, *GRIPPER_DOWN, 0.04],
            ],
            dtype=np.float32,
        )
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])
        self.sequence_duration = float(self.key_times[-1])
        self.grasp_position = self.targets[1, :3].copy()
        self.lift_position = self.targets[3, :3].copy()
        self.minimum_grasp_error = np.inf
        self.maximum_tcp_height = -np.inf

    def _build_ik(self):
        self.n_coords = self.model.joint_coord_count
        self.n_dofs = self.model.joint_dof_count
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.n_coords))
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1
        self.finger_pos_buf = wp.full(1, 0.04, dtype=float, device=self.model.device)
        self.target_joint_q = wp.zeros(self.n_coords, dtype=float, device=self.model.device)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.hand_body,
            link_offset=wp.vec3(0.0, 0.0, 0.107),
            target_positions=wp.array([wp.vec3(*self.targets[0, :3])], dtype=wp.vec3, device=self.model.device),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*self.targets[0, 3:7])], dtype=wp.vec4, device=self.model.device),
        )
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24
        self.tcp_position = wp.zeros(1, dtype=wp.vec3, device=self.model.device)

    def _initialize_robot_pose(self):
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
            device=self.model.device,
        )
        wp.copy(self.target_joint_q, self.ik_joint_q, count=self.n_coords)
        self.state_0.joint_q.assign(self.target_joint_q)
        self.state_0.joint_qd.zero_()
        self.state_1.joint_q.assign(self.target_joint_q)
        self.state_1.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        self.kinematic_contact.update_colliders(self.state_0.body_q, self.state_0.body_qd)

    def update_ik_targets(self):
        t = min(self.sim_time, self.sequence_duration - 1.0e-6)
        interval = int(np.searchsorted(self.key_times, t))
        t_start = self.key_times[interval - 1] if interval > 0 else 0.0
        t_end = self.key_times[interval]
        alpha = float(np.clip((t - t_start) / max(t_end - t_start, 1.0e-6), 0.0, 1.0))
        current = self.targets[interval]
        previous = self.targets[interval - 1] if interval > 0 else current
        target = (1.0 - alpha) * previous + alpha * current

        self.pos_obj.set_target_position(0, wp.vec3(*target[:3]))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target[3:7]))
        self.finger_pos_buf.fill_(float(target[-1]))

    def capture(self):
        self.graph = None
        if self.use_graph:
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
                self.simulate()
            if capture.graph is None:
                raise RuntimeError(f"Graph capture failed on device {self.model.device}")
            self.graph = capture.graph

    def simulate(self):
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
            device=self.model.device,
        )
        wp.copy(self.target_joint_q, self.ik_joint_q, count=self.n_coords)
        wp.launch(
            compute_joint_qd,
            dim=self.n_dofs,
            inputs=[self.target_joint_q, self.state_0.joint_q, self.target_joint_qd, 1.0 / self.frame_dt],
            device=self.model.device,
        )
        self.state_0.joint_q.assign(self.target_joint_q)
        self.state_0.joint_qd.assign(self.target_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.kinematic_contact.update_colliders(self.state_0.body_q, self.state_0.body_qd)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.state_0.assign(self.state_1)

    def _update_motion_metrics(self):
        if not self.collect_metrics:
            return
        wp.launch(
            compute_tcp_position,
            dim=1,
            inputs=[self.state_0.body_q, self.hand_body, wp.vec3(0.0, 0.0, 0.107), self.tcp_position],
            device=self.model.device,
        )
        tcp_position = self.tcp_position.numpy()[0]
        self.minimum_grasp_error = min(
            self.minimum_grasp_error,
            float(np.linalg.norm(tcp_position - self.grasp_position)),
        )
        if self.sim_time >= self.key_times[2]:
            self.maximum_tcp_height = max(self.maximum_tcp_height, float(tcp_position[2]))

        contact_count = 0
        overflow_count = 0
        for contacts in (
            self.kinematic_contact.cloth_vertex_face_contacts,
            self.kinematic_contact.rigid_vertex_face_contacts,
            self.kinematic_contact.edge_edge_contacts,
        ):
            contact_count += min(int(contacts.count.numpy()[0]), contacts.capacity)
            overflow_count += int(contacts.overflow_count.numpy()[0])
        self.maximum_contact_count = max(self.maximum_contact_count, contact_count)
        self.maximum_overflow_count = max(self.maximum_overflow_count, overflow_count)

        centroid_height = float(self.state_0.particle_q.numpy()[:, 2].mean())
        if self.sim_time >= self.key_times[2] and self.pre_lift_centroid_height is None:
            self.pre_lift_centroid_height = centroid_height
        if self.pre_lift_centroid_height is not None:
            lift = centroid_height - self.pre_lift_centroid_height
            self.maximum_cloth_lift = max(self.maximum_cloth_lift, lift)
            if self.key_times[3] <= self.sim_time < self.key_times[4]:
                if lift > 0.10 and contact_count > 0:
                    self.captured_hold_duration += self.frame_dt
                else:
                    self.captured_hold_duration = 0.0

    def step(self):
        self.update_ik_targets()
        if self.graph is not None:
            with wp.ScopedDevice(self.model.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self._update_motion_metrics()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        joint_q = self.state_0.joint_q.numpy()
        particle_q = self.state_0.particle_q.numpy()
        if not np.isfinite(body_q).all() or not np.isfinite(joint_q).all() or not np.isfinite(particle_q).all():
            raise AssertionError("LIMX Franka scene contains non-finite state")

    def test_final(self):
        self.test_post_step()
        if self.minimum_grasp_error >= 0.03:
            raise AssertionError(f"Franka TCP missed the cloth grasp pose by {self.minimum_grasp_error:.4f} m")
        minimum_lift_height = float(self.grasp_position[2]) + 0.10
        if self.maximum_tcp_height <= minimum_lift_height:
            raise AssertionError(
                f"Franka TCP reached only {self.maximum_tcp_height:.4f} m after closing; "
                f"expected more than {minimum_lift_height:.4f} m"
            )
        if self.maximum_contact_count <= 0:
            raise AssertionError("Franka boxes never contacted the LIMX cloth")
        if self.maximum_overflow_count > 0:
            raise AssertionError(f"Franka box contacts overflowed by {self.maximum_overflow_count}")
        if self.maximum_cloth_lift <= 0.10:
            raise AssertionError(f"Franka lifted the cloth centroid by only {self.maximum_cloth_lift:.4f} m")
        if self.captured_hold_duration < 0.5:
            raise AssertionError(f"Franka held the raised cloth for only {self.captured_hold_duration:.3f} s")

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
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable CUDA graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=640)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
