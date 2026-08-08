# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grasp the center of flat LIMX cloth with a top-down Franka."""

import math

import numpy as np
import warp as wp

import newton.examples
from newton.examples.cloth.example_cloth_limx_franka import (
    CLOTH_HEIGHT as SIDE_CLOTH_HEIGHT,
)
from newton.examples.cloth.example_cloth_limx_franka import (
    FPS,
    TABLE_TOP_Z,
)
from newton.examples.cloth.example_cloth_limx_franka import (
    Example as SideGraspExample,
)

FRANKA_BASE = (-0.3, -1.15, -0.1)
CLOTH_CENTER = (0.0, -0.5)
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)
APPROACH_HEIGHT = 0.35
GRASP_HEIGHT = 0.211
LIFT_HEIGHT = 0.42
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0029
SEQUENCE_DURATION = 5.9
CLOTH_HEIGHT = SIDE_CLOTH_HEIGHT


def _create_top_grasp_poses() -> np.ndarray:
    x, y = CLOTH_CENTER
    qx, qy, qz, qw = GRIPPER_DOWN
    return np.asarray(
        [
            [0.5, x, y, APPROACH_HEIGHT, qx, qy, qz, qw, GRIPPER_OPEN],
            [0.8, x, y, GRASP_HEIGHT, qx, qy, qz, qw, GRIPPER_OPEN],
            [0.8, x, y, GRASP_HEIGHT, qx, qy, qz, qw, GRIPPER_CLOSED],
            [0.4, x, y, GRASP_HEIGHT, qx, qy, qz, qw, GRIPPER_CLOSED],
            [1.2, x, y, LIFT_HEIGHT, qx, qy, qz, qw, GRIPPER_CLOSED],
            [0.8, x, y, LIFT_HEIGHT, qx, qy, qz, qw, GRIPPER_CLOSED],
            [0.8, x, y, LIFT_HEIGHT, qx, qy, qz, qw, GRIPPER_OPEN],
            [0.6, x, y, APPROACH_HEIGHT, qx, qy, qz, qw, GRIPPER_OPEN],
        ],
        dtype=np.float32,
    )


class Example(SideGraspExample):
    def __init__(self, viewer, args=None):
        super().__init__(
            viewer,
            args,
            franka_base=FRANKA_BASE,
            cloth_center=CLOTH_CENTER,
            initial_ik_solve_batches=40,
        )

        distance_from_center = np.abs(self.cloth_rest_positions[:, :2] - np.asarray(CLOTH_CENTER))
        self.center_patch_indices = np.flatnonzero(np.all(distance_from_center < 0.04, axis=1))
        self.initial_center_patch_height = float(self.cloth_rest_positions[self.center_patch_indices, 2].mean())
        self.maximum_center_patch_lift = 0.0
        self.minimum_finger_height = np.inf

        self.viewer.set_camera(wp.vec3(0.85, -1.35, 0.9), -25.0, 130.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(0.0, -0.5, 0.28))

    def _build_keyframes(self):
        poses = _create_top_grasp_poses()
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])
        self.sequence_duration = float(self.key_times[-1])
        self.descend_end_time = float(self.key_times[1])
        self.preclose_time = self.descend_end_time
        self.close_time = float(self.key_times[2])
        self.lift_time = float(self.key_times[4])
        self.hold_end_time = float(self.key_times[5])
        self.grasp_position = self.targets[1, :3].copy()
        self.lift_position = self.targets[4, :3].copy()
        self.minimum_grasp_error = np.inf
        self.maximum_tcp_height = -np.inf

    def _update_motion_metrics(self):
        super()._update_motion_metrics()
        if not self.collect_metrics:
            return

        particle_positions = self.state_0.particle_q.numpy()
        center_patch_height = float(particle_positions[self.center_patch_indices, 2].mean())
        self.maximum_center_patch_lift = max(
            self.maximum_center_patch_lift,
            center_patch_height - self.initial_center_patch_height,
        )
        finger_positions = self.gripper_contact.collider_positions.numpy()
        self.minimum_finger_height = min(self.minimum_finger_height, float(finger_positions[:, 2].min()))

    def test_final(self):
        self.test_post_step()
        if self.minimum_grasp_error >= 0.01:
            raise AssertionError(f"Franka TCP missed the top grasp pose by {self.minimum_grasp_error:.4f} m")
        if self.maximum_ccd_binding_count <= 0:
            raise AssertionError("Top-down Franka motion never produced a cloth-vertex CCD binding")
        if self.maximum_overflow_count > 0:
            raise AssertionError(f"Top-down Franka contacts overflowed by {self.maximum_overflow_count}")
        if self.minimum_finger_height <= TABLE_TOP_Z + 0.004:
            raise AssertionError(
                f"Franka finger box reached {self.minimum_finger_height:.4f} m and entered the table clearance"
            )
        if self.maximum_center_patch_lift <= 0.08:
            raise AssertionError(f"Franka lifted the center cloth patch by only {self.maximum_center_patch_lift:.4f} m")


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=math.ceil(SEQUENCE_DURATION * FPS) + 10)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
