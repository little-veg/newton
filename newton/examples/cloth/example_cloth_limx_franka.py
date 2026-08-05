# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Place a fixed square LIMX cloth on a table for a Franka grasp sequence."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.solvers import SolverFeatherstone

FPS = 60
SIM_SUBSTEPS = 10
TABLE_CENTER = (0.0, -0.5, 0.1)
TABLE_HALF_EXTENTS = (0.4, 0.4, 0.1)
TABLE_TOP_Z = 0.2
CLOTH_CENTER = (0.0, -0.5)
CLOTH_WIDTH = 0.4
CLOTH_GRID_CELLS = 20
CLOTH_HEIGHT = TABLE_TOP_Z + 0.005
CLOTH_PARTICLE_RADIUS = 0.003
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
SEQUENCE_DURATION = 5.8


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

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*TABLE_CENTER), wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
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
            mass=[0.2 / particle_count] * particle_count,
            radius=[CLOTH_PARTICLE_RADIUS] * particle_count,
        )
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        builder.add_ground_plane()
        builder.color()

        self.model = builder.finalize(requires_grad=False)
        self.hand_body = _find_label_index(self.model.body_label, "fr3_hand")
        self.cloth_particle_indices = np.arange(self.model.particle_count, dtype=np.int32)
        self.inverse_rest_matrices = self.model.tri_poses.numpy()
        edge_rows = newton.utils.MeshAdjacency(triangles).edge_indices
        self.dihedral_indices = edge_rows[edge_rows[:, 1] >= 0][:, [2, 3, 0, 1]]

        flags = self.model.particle_flags.numpy()
        flags &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(
            flags,
            dtype=self.model.particle_flags.dtype,
            device=self.model.device,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.gravity_earth = wp.array([wp.vec3(0.0, 0.0, -9.81)], dtype=wp.vec3, device=self.model.device)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self._build_keyframes()
        self._build_ik()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.85, -1.45, 0.95), -22.0, 125.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(0.0, -0.5, 0.25))

        self.use_graph = self.use_graph and self.model.device.is_cuda
        self.capture()

    def _build_keyframes(self):
        poses = np.asarray(
            [
                [1.5, 0.0, -0.5, 0.38, *GRIPPER_DOWN, 0.04],
                [1.2, 0.0, -0.5, CLOTH_HEIGHT, *GRIPPER_DOWN, 0.04],
                [0.8, 0.0, -0.5, CLOTH_HEIGHT, *GRIPPER_DOWN, 0.0],
                [1.5, 0.0, -0.5, 0.38, *GRIPPER_DOWN, 0.0],
                [0.8, 0.0, -0.5, 0.38, *GRIPPER_DOWN, 0.04],
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

        particle_count = self.model.particle_count
        self.model.particle_count = 0
        self.model.gravity.assign(self.gravity_zero)
        try:
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.state_1.clear_forces()
                self.state_0.joint_qd.assign(self.target_joint_qd)
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
        finally:
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

    def _update_motion_metrics(self):
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
        particle_q = self.state_0.particle_q.numpy()
        np.testing.assert_allclose(particle_q, self.cloth_rest_positions, rtol=0.0, atol=1.0e-7)
        if self.minimum_grasp_error >= 0.03:
            raise AssertionError(f"Franka TCP missed the cloth grasp pose by {self.minimum_grasp_error:.4f} m")
        minimum_lift_height = float(self.grasp_position[2]) + 0.10
        if self.maximum_tcp_height <= minimum_lift_height:
            raise AssertionError(
                f"Franka TCP reached only {self.maximum_tcp_height:.4f} m after closing; "
                f"expected more than {minimum_lift_height:.4f} m"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
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
    parser.set_defaults(num_frames=360)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
