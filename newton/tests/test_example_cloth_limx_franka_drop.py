# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxFrankaDrop(unittest.TestCase):
    def test_scene_drops_active_cloth_onto_kinematic_gripper(self):
        """Drop active cloth toward selected Franka wrist and gripper collision meshes."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka_drop")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=20), SimpleNamespace(graph_capture=False))
            for _ in range(20):
                example.step()
                example.test_post_step()

        flags = example.model.particle_flags.numpy()
        collider_bodies = {
            example.model.body_label[int(example.model.shape_body.numpy()[shape])]
            for shape in example.collider_shape_indices
        }
        self.assertEqual(example.model.particle_count, 441)
        self.assertTrue(np.all((flags & int(newton.ParticleFlags.ACTIVE)) != 0))
        self.assertTrue(all(int(example.model.shape_body.numpy()[shape]) >= 0 for shape in example.collider_shape_indices))
        self.assertTrue(any(label.endswith("fr3_hand") for label in collider_bodies))
        self.assertTrue(any(label.endswith("fr3_leftfinger") for label in collider_bodies))
        self.assertTrue(any(label.endswith("fr3_rightfinger") for label in collider_bodies))
        self.assertGreater(example.maximum_contact_count, 0)
        self.assertEqual(example.maximum_overflow_count, 0)


if __name__ == "__main__":
    unittest.main()
