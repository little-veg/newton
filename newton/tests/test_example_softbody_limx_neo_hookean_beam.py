# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LIMX Neo-Hookean cantilever example."""

import argparse
import unittest

import numpy as np
import warp as wp

from newton.examples.softbody.example_softbody_limx_neo_hookean_beam import Example
from newton.viewer import ViewerNull


def make_args(*, line_search: bool):
    return argparse.Namespace(
        line_search=line_search,
        convergence_study=False,
        output_directory=".",
    )


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestNeoHookeanBeamExample(unittest.TestCase):
    def test_builds_approved_cantilever_configuration(self):
        """Build the approved undamped logarithmic Neo-Hookean cantilever."""
        with wp.ScopedDevice("cuda:0"):
            example = Example(ViewerNull(num_frames=1), make_args(line_search=True))

        self.assertEqual(example.grid_dimensions, (12, 2, 2))
        self.assertAlmostEqual(example.frame_dt, 0.01)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertIsNotNone(example.solver.line_search)
        self.assertEqual(example.model.shape_count, 0)
        self.assertEqual(len(example.anchor_indices), 9)

    def test_one_step_keeps_state_finite_and_positive(self):
        """Advance one CUDA step without inversion or non-finite state."""
        with wp.ScopedDevice("cuda:0"):
            example = Example(ViewerNull(num_frames=1), make_args(line_search=True))
            example.step()
            example.test_post_step()
            positions = example.state_0.particle_q.numpy()
            velocities = example.state_0.particle_qd.numpy()

        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(velocities).all())

    def test_free_end_falls_while_left_face_stays_anchored(self):
        """Drop the free cantilever end while retaining the anchored face."""
        with wp.ScopedDevice("cuda:0"):
            example = Example(ViewerNull(num_frames=20), make_args(line_search=True))
            for _ in range(20):
                example.step()
                example.test_post_step()
            example.test_final()

    def test_no_line_search_selects_full_steps(self):
        """Disable Armijo explicitly for the visual comparison variant."""
        with wp.ScopedDevice("cuda:0"):
            example = Example(ViewerNull(num_frames=1), make_args(line_search=False))

        self.assertIsNone(example.solver.line_search)


if __name__ == "__main__":
    unittest.main(verbosity=2)
