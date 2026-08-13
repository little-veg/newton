# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Mobile ALOHA Dual-Arm IK
#
# Downloads the complete Split Mobile ALOHA model from AgileX Robotics at
# a pinned revision. The upstream assets are cached locally and are not
# distributed with Newton.
#
# Command: python -m newton.examples robot_mobile_aloha
#
###########################################################################

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

MOBILE_ALOHA_URL = "https://github.com/agilexrobotics/mobile_aloha_sim.git"
MOBILE_ALOHA_REF = "594da182508f0780a1a81a40494552564babec93"
MOBILE_ALOHA_URDF = Path("split_aloha_mid_360/urdf/split_aloha_mid_360_with_piper.urdf")
MOBILE_ALOHA_PACKAGES = ("split_aloha_mid_360", "piper_description")
LOCKED_JOINT_NAMES = (
    "fr_steering_joint",
    "fr_wheel",
    "fl_steering_joint",
    "fl_wheel",
    "rl_steering_joint",
    "rl_wheel",
    "rr_steering_joint",
    "rr_wheel",
    "lifting_joint",
)
MODEL_LABEL = "split_aloha_mid_360_with_piper"
ARM_JOINT_NAMES = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in range(1, 7))
FINGER_JOINT_NAMES = tuple(f"{side}/joint{joint}" for side in ("left", "right") for joint in (7, 8))
TCP_BODY_NAMES = ("left/link6", "right/link6")
TCP_OFFSET = wp.vec3(0.0, 0.0, 0.13503)


def resolve_mobile_aloha_asset_root(asset_root: str | Path | None) -> Path:
    """Resolve and validate the Mobile ALOHA repository root."""
    if asset_root is None:
        try:
            root = newton.utils.download_git_folder(MOBILE_ALOHA_URL, ".", ref=MOBILE_ALOHA_REF)
        except Exception as error:
            raise RuntimeError(
                f"Unable to download Mobile ALOHA assets from {MOBILE_ALOHA_URL} at {MOBILE_ALOHA_REF}. "
                "Download that revision manually and pass --asset-root PATH."
            ) from error
    else:
        root = Path(asset_root).expanduser()

    root = root.resolve()
    for package in MOBILE_ALOHA_PACKAGES:
        package_path = root / package
        if not package_path.is_dir():
            raise FileNotFoundError(f"Mobile ALOHA package directory is missing: {package_path}")
    urdf_path = root / MOBILE_ALOHA_URDF
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Mobile ALOHA URDF is missing: {urdf_path}")
    return root


def normalize_mobile_aloha_urdf(asset_root: str | Path) -> str:
    """Return a fixed-base Mobile ALOHA URDF with absolute mesh paths."""
    root_path = resolve_mobile_aloha_asset_root(asset_root)
    urdf_root = ET.parse(root_path / MOBILE_ALOHA_URDF).getroot()

    joints_by_name: dict[str, list[ET.Element]] = {}
    for joint in urdf_root.findall("joint"):
        name = joint.get("name")
        if name is not None:
            joints_by_name.setdefault(name, []).append(joint)

    for name in LOCKED_JOINT_NAMES:
        matches = joints_by_name.get(name, [])
        if not matches:
            raise ValueError(f"Required Mobile ALOHA joint '{name}' is missing")
        if len(matches) > 1:
            raise ValueError(f"Required Mobile ALOHA joint '{name}' is duplicate")
        matches[0].set("type", "fixed")

    for mesh in urdf_root.iter("mesh"):
        filename = mesh.get("filename")
        if filename is None or not filename.startswith("package://"):
            continue
        package_uri = filename.removeprefix("package://")
        package, separator, relative_name = package_uri.partition("/")
        if not separator or package not in MOBILE_ALOHA_PACKAGES:
            raise ValueError(f"Unsupported Mobile ALOHA package URI: {filename}")
        package_root = (root_path / package).resolve()
        mesh_path = (package_root / relative_name).resolve()
        if not mesh_path.is_relative_to(package_root):
            raise ValueError(f"Mobile ALOHA mesh URI escapes package root: {filename}")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mobile ALOHA mesh is missing: {mesh_path}")
        mesh.set("filename", str(mesh_path))

        scale_text = mesh.get("scale")
        if mesh_path.suffix.lower() == ".dae" and scale_text is not None:
            scale = np.asarray([float(value) for value in scale_text.split()], dtype=np.float64)
            collada_root = ET.parse(mesh_path).getroot()
            unit = collada_root.find("{*}asset/{*}unit")
            if unit is not None and unit.get("meter") is not None:
                meter = float(unit.get("meter"))
                if scale.shape == (3,) and meter > 0.0 and np.allclose(scale * meter, 1.0):
                    mesh.set("scale", " ".join(f"{value:g}" for value in scale * meter))

    return ET.tostring(urdf_root, encoding="unicode")


def find_unique_label(labels: Sequence[str], required: str) -> int:
    """Return the index of one required exact label."""
    matches = [index for index, label in enumerate(labels) if label == required]
    if not matches:
        raise ValueError(f"Required label '{required}' is missing")
    if len(matches) > 1:
        raise ValueError(f"Required label '{required}' is duplicate")
    return matches[0]


def clamp_and_rate_limit_targets(
    solution: np.ndarray,
    previous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity: np.ndarray,
    frame_dt: float,
) -> np.ndarray:
    """Clamp finite joint targets to coordinate and per-frame velocity limits."""
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (solution, previous, lower, upper, velocity))
    solution_array, previous_array, lower_array, upper_array, velocity_array = arrays
    if any(value.shape != previous_array.shape for value in arrays):
        raise ValueError("Target filter arrays must have identical shapes")
    if not np.isfinite(frame_dt) or frame_dt <= 0.0:
        raise ValueError("frame_dt must be finite and positive")
    if not np.all(np.isfinite(velocity_array)) or np.any(velocity_array < 0.0):
        raise ValueError("velocity limits must be finite and nonnegative")
    if np.any(lower_array > upper_array):
        raise ValueError("lower limits must not exceed upper limits")
    if not np.all(np.isfinite(solution_array)):
        return previous_array.copy()

    bounded = np.clip(solution_array, lower_array, upper_array)
    max_increment = velocity_array * frame_dt
    return previous_array + np.clip(bounded - previous_array, -max_increment, max_increment)


def gripper_joint_targets(opening: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map a total gripper opening to two bounded opposite coordinates."""
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if lower_array.shape != (2,) or upper_array.shape != (2,):
        raise ValueError("Gripper limits must each contain two coordinates")
    if not np.isfinite(opening) or opening < 0.0:
        raise ValueError("Gripper opening must be finite and nonnegative")
    if np.any(lower_array > upper_array):
        raise ValueError("Gripper lower limits must not exceed upper limits")
    desired = np.array((0.5 * opening, -0.5 * opening), dtype=np.float64)
    return np.clip(desired, lower_array, upper_array)


def _qualified_label(name: str) -> str:
    return f"{MODEL_LABEL}/{name}"


def _tcp_transform(body_transform: np.ndarray) -> wp.transform:
    link_transform = wp.transform(*body_transform.tolist())
    return wp.transform_multiply(link_transform, wp.transform(TCP_OFFSET, wp.quat_identity()))


def _rotation_error(actual: wp.transform, target: wp.transform) -> float:
    actual_q = np.asarray(wp.transform_get_rotation(actual), dtype=np.float64)
    target_q = np.asarray(wp.transform_get_rotation(target), dtype=np.float64)
    cosine = float(np.clip(abs(np.dot(actual_q, target_q)), 0.0, 1.0))
    return 2.0 * float(np.arccos(cosine))


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.test_mode = bool(getattr(args, "test", False))
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.ik_iterations = 24
        self.gripper_openings = [0.07, 0.07]

        asset_root = resolve_mobile_aloha_asset_root(getattr(args, "asset_root", None))
        normalized_urdf = normalize_mobile_aloha_urdf(asset_root)
        velocity_by_joint = {}
        for joint in ET.fromstring(normalized_urdf).findall("joint"):
            limit = joint.find("limit")
            if limit is not None and limit.get("velocity") is not None:
                velocity_by_joint[joint.get("name")] = float(limit.get("velocity"))

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_urdf(
            normalized_urdf,
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        root_joint_label = _qualified_label("fixed_base")
        find_unique_label(builder.joint_label, root_joint_label)
        builder.collapse_fixed_joints(joints_to_keep=[root_joint_label])

        arm_pose = (0.0, 1.2, -1.2, 0.0, 0.0, 0.0)
        arm_gains = (10000.0, 2000.0, 2000.0, 500.0, 200.0, 200.0)
        arm_damping = (500.0, 5.0, 20.0, 5.0, 5.0, 5.0)
        arm_joint_indices = []
        arm_coord_indices = []
        arm_dof_indices = []
        for joint_name in ARM_JOINT_NAMES:
            joint_index = find_unique_label(builder.joint_label, _qualified_label(joint_name))
            if builder.joint_type[joint_index] != newton.JointType.REVOLUTE:
                raise ValueError(f"Mobile ALOHA arm joint is not revolute: {joint_name}")
            joint_number = int(joint_name.rsplit("joint", 1)[1]) - 1
            coord_index = builder.joint_q_start[joint_index]
            dof_index = builder.joint_qd_start[joint_index]
            builder.joint_q[coord_index] = arm_pose[joint_number]
            builder.joint_target_q[coord_index] = arm_pose[joint_number]
            builder.joint_target_ke[dof_index] = arm_gains[joint_number]
            builder.joint_target_kd[dof_index] = 0.0
            builder.joint_damping[dof_index] = arm_damping[joint_number]
            builder.joint_target_mode[dof_index] = int(newton.JointTargetMode.POSITION)
            builder.joint_velocity_limit[dof_index] = velocity_by_joint[joint_name]
            arm_joint_indices.append(joint_index)
            arm_coord_indices.append(coord_index)
            arm_dof_indices.append(dof_index)

        finger_joint_indices = []
        finger_coord_indices = []
        finger_dof_indices = []
        for joint_name in FINGER_JOINT_NAMES:
            joint_index = find_unique_label(builder.joint_label, _qualified_label(joint_name))
            if builder.joint_type[joint_index] != newton.JointType.PRISMATIC:
                raise ValueError(f"Mobile ALOHA finger joint is not prismatic: {joint_name}")
            coord_index = builder.joint_q_start[joint_index]
            dof_index = builder.joint_qd_start[joint_index]
            initial_position = 0.035 if joint_name.endswith("joint7") else -0.035
            builder.joint_q[coord_index] = initial_position
            builder.joint_target_q[coord_index] = initial_position
            builder.joint_target_ke[dof_index] = 10000.0
            builder.joint_target_kd[dof_index] = 0.0
            builder.joint_damping[dof_index] = 100.0
            builder.joint_target_mode[dof_index] = int(newton.JointTargetMode.POSITION)
            builder.joint_velocity_limit[dof_index] = velocity_by_joint[joint_name]
            finger_joint_indices.append(joint_index)
            finger_coord_indices.append(coord_index)
            finger_dof_indices.append(dof_index)

        self.arm_joint_indices = np.asarray(arm_joint_indices, dtype=np.int32)
        self.arm_coord_indices = np.asarray(arm_coord_indices, dtype=np.int32)
        self.arm_dof_indices = np.asarray(arm_dof_indices, dtype=np.int32)
        self.finger_joint_indices = np.asarray(finger_joint_indices, dtype=np.int32)
        self.finger_coord_indices = np.asarray(finger_coord_indices, dtype=np.int32)
        self.finger_dof_indices = np.asarray(finger_dof_indices, dtype=np.int32)
        self.root_body_index = find_unique_label(builder.body_label, _qualified_label("base_link"))
        self.tcp_body_indices = tuple(
            find_unique_label(builder.body_label, _qualified_label(name)) for name in TCP_BODY_NAMES
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.control.joint_target_qd.zero_()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=True,
            solver="newton",
            integrator="implicitfast",
            use_mujoco_cpu=False,
        )

        body_transforms = self.state_0.body_q.numpy()
        self.root_initial_transform = body_transforms[self.root_body_index].copy()
        self.actual_tcp_transforms = [
            _tcp_transform(body_transforms[body_index]) for body_index in self.tcp_body_indices
        ]
        self.target_tcp_transforms = list(self.actual_tcp_transforms)
        if self.test_mode:
            for index, transform in enumerate(self.target_tcp_transforms):
                position = wp.transform_get_translation(transform)
                rotation = wp.transform_get_rotation(transform)
                self.target_tcp_transforms[index] = wp.transform(
                    wp.vec3(position[0], position[1], position[2] + 0.02), rotation
                )

        self.position_objectives = []
        self.rotation_objectives = []
        objectives = []
        for body_index, transform in zip(self.tcp_body_indices, self.target_tcp_transforms, strict=True):
            position_objective = ik.IKObjectivePosition(
                link_index=body_index,
                link_offset=TCP_OFFSET,
                target_positions=wp.array(
                    [wp.transform_get_translation(transform)], dtype=wp.vec3, device=self.model.device
                ),
            )
            rotation = wp.transform_get_rotation(transform)
            rotation_objective = ik.IKObjectiveRotation(
                link_index=body_index,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array(
                    [wp.vec4(rotation[0], rotation[1], rotation[2], rotation[3])],
                    dtype=wp.vec4,
                    device=self.model.device,
                ),
            )
            self.position_objectives.append(position_objective)
            self.rotation_objectives.append(rotation_objective)
            objectives.extend((position_objective, rotation_objective))
        objectives.append(
            ik.IKObjectiveJointLimit(
                joint_limit_lower=self.model.joint_limit_lower,
                joint_limit_upper=self.model.joint_limit_upper,
                weight=10.0,
            )
        )

        joint_dof_mask = np.zeros(self.model.joint_dof_count, dtype=bool)
        joint_dof_mask[self.arm_dof_indices] = True
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=objectives,
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            sampler=ik.IKSampler.NONE,
            lambda_initial=0.1,
            joint_dof_mask=wp.array(joint_dof_mask, dtype=wp.bool, device=self.model.device),
        )
        initial_joint_q = self.model.joint_q.numpy().astype(np.float32)
        self.last_valid_ik_q = initial_joint_q.copy()
        self.joint_q_ik = wp.array(initial_joint_q[None, :], dtype=wp.float32, device=self.model.device)

        joint_limit_lower = self.model.joint_limit_lower.numpy()
        joint_limit_upper = self.model.joint_limit_upper.numpy()
        self.arm_lower_limits = joint_limit_lower[self.arm_dof_indices]
        self.arm_upper_limits = joint_limit_upper[self.arm_dof_indices]
        self.arm_velocity_limits = self.model.joint_velocity_limit.numpy()[self.arm_dof_indices]
        self.finger_lower_limits = joint_limit_lower[self.finger_dof_indices]
        self.finger_upper_limits = joint_limit_upper[self.finger_dof_indices]
        self.control_target_q = self.control.joint_target_q.numpy()
        self.previous_arm_targets = initial_joint_q[self.arm_coord_indices].astype(np.float64)
        self.max_arm_target_increments = np.zeros(len(self.arm_coord_indices), dtype=np.float64)
        self.tcp_position_errors = [0.02 if self.test_mode else 0.0, 0.02 if self.test_mode else 0.0]
        self.tcp_rotation_errors = [0.0, 0.0]

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.0, -2.4, 1.7), -16.0, 126.1)
        self.capture()

    def capture(self):
        self.graph_ik = None
        self.graph_sim = None
        if not self.model.device.is_cuda:
            return

        with wp.ScopedCapture(device=self.model.device) as capture:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iterations)
        self.graph_ik = capture.graph

        with wp.ScopedCapture(device=self.model.device) as capture:
            self.simulate()
        self.graph_sim = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _push_ik_targets(self):
        for index, transform in enumerate(self.target_tcp_transforms):
            self.position_objectives[index].set_target_position(0, wp.transform_get_translation(transform))
            rotation = wp.transform_get_rotation(transform)
            self.rotation_objectives[index].set_target_rotation(
                0, wp.vec4(rotation[0], rotation[1], rotation[2], rotation[3])
            )

    def _update_commands(self):
        self._push_ik_targets()
        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iterations)
        ik_result = self.joint_q_ik.numpy()[0]
        arm_solution = ik_result[self.arm_coord_indices]
        if np.all(np.isfinite(arm_solution)):
            self.last_valid_ik_q = ik_result.copy()
        else:
            self.joint_q_ik.assign(self.last_valid_ik_q[None, :])

        arm_targets = clamp_and_rate_limit_targets(
            arm_solution,
            self.previous_arm_targets,
            self.arm_lower_limits,
            self.arm_upper_limits,
            self.arm_velocity_limits,
            self.frame_dt,
        )
        increments = np.abs(arm_targets - self.previous_arm_targets)
        self.max_arm_target_increments = np.maximum(self.max_arm_target_increments, increments)
        self.previous_arm_targets = arm_targets

        target_q = self.control_target_q
        target_q[self.arm_coord_indices] = arm_targets
        for side in range(2):
            finger_slice = slice(2 * side, 2 * side + 2)
            coord_indices = self.finger_coord_indices[finger_slice]
            target_q[coord_indices] = gripper_joint_targets(
                self.gripper_openings[side],
                self.finger_lower_limits[finger_slice],
                self.finger_upper_limits[finger_slice],
            )
        self.control.joint_target_q.assign(target_q)

    def _update_tcp_errors(self):
        body_transforms = self.state_0.body_q.numpy()
        self.actual_tcp_transforms = [
            _tcp_transform(body_transforms[body_index]) for body_index in self.tcp_body_indices
        ]
        for index, (actual, target) in enumerate(
            zip(self.actual_tcp_transforms, self.target_tcp_transforms, strict=True)
        ):
            actual_position = np.asarray(wp.transform_get_translation(actual), dtype=np.float64)
            target_position = np.asarray(wp.transform_get_translation(target), dtype=np.float64)
            self.tcp_position_errors[index] = float(np.linalg.norm(actual_position - target_position))
            self.tcp_rotation_errors[index] = _rotation_error(actual, target)

    def step(self):
        self._update_commands()
        if self.graph_sim is not None:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._update_tcp_errors()

    def test_post_step(self):
        arrays = (
            self.state_0.joint_q.numpy(),
            self.state_0.joint_qd.numpy(),
            self.state_0.body_q.numpy(),
            self.control.joint_target_q.numpy(),
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise AssertionError("Mobile ALOHA dynamic state contains non-finite values")

    def test_final(self):
        joint_types = self.model.joint_type.numpy()
        if len(self.arm_dof_indices) != 12 or not np.all(
            joint_types[self.arm_joint_indices] == int(newton.JointType.REVOLUTE)
        ):
            raise AssertionError("Mobile ALOHA must contain exactly twelve revolute arm DoFs")
        if len(self.finger_dof_indices) != 4 or not np.all(
            joint_types[self.finger_joint_indices] == int(newton.JointType.PRISMATIC)
        ):
            raise AssertionError("Mobile ALOHA must contain exactly four prismatic finger DoFs")
        if self.model.joint_dof_count != 16:
            raise AssertionError(f"Mobile ALOHA has {self.model.joint_dof_count} active DoFs instead of 16")

        joint_q = self.state_0.joint_q.numpy()
        arm_q = joint_q[self.arm_coord_indices]
        if np.any(arm_q < self.arm_lower_limits - 1.0e-5) or np.any(arm_q > self.arm_upper_limits + 1.0e-5):
            raise AssertionError("Mobile ALOHA arm coordinates exceeded their limits")
        if max(self.tcp_position_errors) >= 0.02:
            raise AssertionError(f"Mobile ALOHA TCP position errors are too large: {self.tcp_position_errors}")
        if max(self.tcp_rotation_errors) >= 0.10:
            raise AssertionError(f"Mobile ALOHA TCP rotation errors are too large: {self.tcp_rotation_errors}")

        final_root = self.state_0.body_q.numpy()[self.root_body_index]
        initial_root = self.root_initial_transform
        root_position_drift = float(np.linalg.norm(final_root[:3] - initial_root[:3]))
        root_rotation_drift = _rotation_error(wp.transform(*final_root), wp.transform(*initial_root))
        if root_position_drift >= 1.0e-6 or root_rotation_drift >= 1.0e-6:
            raise AssertionError("Mobile ALOHA fixed root moved")

        for side in range(2):
            finger_q = joint_q[self.finger_coord_indices[2 * side : 2 * side + 2]]
            if abs(float(np.sum(finger_q))) >= 1.0e-5:
                raise AssertionError(f"Mobile ALOHA finger pair {side} is not symmetric: {finger_q}")
        max_allowed_increment = self.arm_velocity_limits * self.frame_dt + 1.0e-5
        if np.any(self.max_arm_target_increments > max_allowed_increment):
            raise AssertionError("Mobile ALOHA arm targets exceeded their velocity limits")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_gizmo("left_tcp_target", self.target_tcp_transforms[0])
        self.viewer.log_gizmo("right_tcp_target", self.target_tcp_transforms[1])
        self.viewer.end_frame()

    def gui(self, ui):
        _changed, self.gripper_openings[0] = ui.slider_float("Left opening [m]", self.gripper_openings[0], 0.0, 0.1)
        _changed, self.gripper_openings[1] = ui.slider_float("Right opening [m]", self.gripper_openings[1], 0.0, 0.1)
        ui.separator()
        ui.text(f"Left TCP: {self.tcp_position_errors[0]:.4f} m, {self.tcp_rotation_errors[0]:.4f} rad")
        ui.text(f"Right TCP: {self.tcp_position_errors[1]:.4f} m, {self.tcp_rotation_errors[1]:.4f} rad")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--asset-root",
            type=Path,
            default=None,
            help="Path to a Mobile ALOHA checkout containing both ROS packages.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
