# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

from newton.solvers import SolverMuJoCo
from newton.viewer import ViewerNull

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


class TestMobileAlohaHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("newton.examples.robot.example_robot_mobile_aloha")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.asset_root = Path(self.temp_dir.name)
        self.urdf_path = self._write_fixture()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_fixture(self) -> Path:
        split_package = self.asset_root / "split_aloha_mid_360"
        piper_package = self.asset_root / "piper_description"
        urdf_dir = split_package / "urdf"
        split_mesh_dir = split_package / "meshes"
        piper_mesh_dir = piper_package / "meshes"
        urdf_dir.mkdir(parents=True)
        split_mesh_dir.mkdir(parents=True)
        piper_mesh_dir.mkdir(parents=True)
        (split_mesh_dir / "base.dae").write_text("mesh\n", encoding="utf-8")
        (piper_mesh_dir / "link.stl").write_text("mesh\n", encoding="utf-8")

        robot = ET.Element("robot", name="synthetic_mobile_aloha")
        for name in LOCKED_JOINT_NAMES:
            ET.SubElement(robot, "joint", name=name, type="continuous")
        ET.SubElement(robot, "joint", name="left/joint1", type="revolute")
        ET.SubElement(robot, "joint", name="right/joint1", type="revolute")
        left_link = ET.SubElement(robot, "link", name="left/link1")
        ET.SubElement(
            ET.SubElement(left_link, "visual"), "mesh", filename="package://piper_description/meshes/link.stl"
        )
        base_link = ET.SubElement(robot, "link", name="base_link")
        ET.SubElement(
            ET.SubElement(base_link, "visual"),
            "mesh",
            filename="package://split_aloha_mid_360/meshes/base.dae",
        )

        urdf_path = urdf_dir / "split_aloha_mid_360_with_piper.urdf"
        ET.ElementTree(robot).write(urdf_path, encoding="unicode")
        return urdf_path

    def test_resolve_local_asset_root(self):
        """Resolve a complete local Mobile ALOHA asset tree."""
        resolved = self.module.resolve_mobile_aloha_asset_root(self.asset_root)
        self.assertEqual(resolved, self.asset_root.resolve())

    def test_normalize_mobile_aloha_urdf(self):
        """Fix only base mechanisms and resolve package meshes absolutely."""
        source_xml = self.urdf_path.read_text(encoding="utf-8")

        xml = self.module.normalize_mobile_aloha_urdf(self.asset_root)

        root = ET.fromstring(xml)
        types = {joint.get("name"): joint.get("type") for joint in root.findall("joint")}
        self.assertTrue(all(types[name] == "fixed" for name in LOCKED_JOINT_NAMES))
        self.assertEqual(types["left/joint1"], "revolute")
        self.assertEqual(types["right/joint1"], "revolute")
        for mesh in root.iter("mesh"):
            filename = Path(mesh.get("filename"))
            self.assertTrue(filename.is_absolute())
            self.assertTrue(filename.is_file())
        self.assertEqual(self.urdf_path.read_text(encoding="utf-8"), source_xml)

    def test_rejects_incomplete_mobile_aloha_assets(self):
        """Reject missing packages, lock joints, unsupported packages, and meshes."""
        cases = ("missing package", "missing lock", "duplicate lock", "unsupported package", "missing mesh")
        for case in cases:
            with self.subTest(case=case):
                self.temp_dir.cleanup()
                self.temp_dir = tempfile.TemporaryDirectory()
                self.asset_root = Path(self.temp_dir.name)
                self.urdf_path = self._write_fixture()

                if case == "missing package":
                    (self.asset_root / "piper_description").rename(self.asset_root / "piper_description_missing")
                    with self.assertRaisesRegex(FileNotFoundError, "piper_description"):
                        self.module.resolve_mobile_aloha_asset_root(self.asset_root)
                    continue

                root = ET.parse(self.urdf_path).getroot()
                if case == "missing lock":
                    root.remove(next(joint for joint in root.findall("joint") if joint.get("name") == "fr_wheel"))
                    expected_exception = ValueError
                    expected_message = "fr_wheel.*missing"
                elif case == "duplicate lock":
                    ET.SubElement(root, "joint", name="fr_wheel", type="continuous")
                    expected_exception = ValueError
                    expected_message = "fr_wheel.*duplicate"
                elif case == "unsupported package":
                    next(root.iter("mesh")).set("filename", "package://unexpected/meshes/link.stl")
                    expected_exception = ValueError
                    expected_message = "unexpected"
                else:
                    next(root.iter("mesh")).set("filename", "package://piper_description/meshes/missing.stl")
                    expected_exception = FileNotFoundError
                    expected_message = "missing.stl"
                ET.ElementTree(root).write(self.urdf_path, encoding="unicode")

                with self.assertRaisesRegex(expected_exception, expected_message):
                    self.module.normalize_mobile_aloha_urdf(self.asset_root)

    def test_find_unique_label(self):
        """Resolve one exact label and reject missing or duplicate labels."""
        self.assertEqual(self.module.find_unique_label(["a", "left/joint1"], "left/joint1"), 1)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.module.find_unique_label(["a"], "left/joint1")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.module.find_unique_label(["left/joint1", "left/joint1"], "left/joint1")

    def test_clamp_and_rate_limit_targets(self):
        """Clamp arm commands to coordinate and per-frame velocity limits."""
        result = self.module.clamp_and_rate_limit_targets(
            np.array([2.0, -2.0]),
            np.array([0.0, 0.0]),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            np.array([3.0, 6.0]),
            0.1,
        )
        np.testing.assert_allclose(result, [0.3, -0.6])

        stale = self.module.clamp_and_rate_limit_targets(
            np.array([np.nan, 0.0]),
            np.array([0.2, -0.2]),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            np.array([3.0, 6.0]),
            0.1,
        )
        np.testing.assert_array_equal(stale, [0.2, -0.2])

    def test_rejects_invalid_target_filter_inputs(self):
        """Reject mismatched target arrays and invalid rate-limit parameters."""
        valid = np.array([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "shape"):
            self.module.clamp_and_rate_limit_targets(np.array([0.0]), valid, valid - 1.0, valid + 1.0, valid + 1.0, 0.1)
        with self.assertRaisesRegex(ValueError, "frame_dt"):
            self.module.clamp_and_rate_limit_targets(valid, valid, valid - 1.0, valid + 1.0, valid + 1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "velocity"):
            self.module.clamp_and_rate_limit_targets(valid, valid, valid - 1.0, valid + 1.0, valid - 1.0, 0.1)

    def test_gripper_joint_targets(self):
        """Map total opening to opposite finger coordinates within limits."""
        result = self.module.gripper_joint_targets(0.1, np.array([0.0, -0.04]), np.array([0.04, 0.0]))
        np.testing.assert_allclose(result, [0.04, -0.04])


class TestMobileAlohaExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("newton.examples.robot.example_robot_mobile_aloha")

    def test_tracks_both_tcp_targets(self):
        """Track reachable dual-TCP targets through dynamic joint drives."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA is unavailable")
        try:
            SolverMuJoCo.import_mujoco()
        except Exception as error:
            self.skipTest(f"MuJoCo Warp is unavailable: {error}")

        example_type = self.module.Example
        asset_root = self.module.resolve_mobile_aloha_asset_root(None)
        args = types.SimpleNamespace(asset_root=str(asset_root), test=True)
        with wp.ScopedDevice(devices[0]):
            example = example_type(ViewerNull(num_frames=180), args)
            for _ in range(180):
                example.step()
                example.test_post_step()
            example.test_final()


if __name__ == "__main__":
    unittest.main(verbosity=2)
