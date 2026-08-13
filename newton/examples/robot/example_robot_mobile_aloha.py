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

from collections.abc import Sequence
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

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
