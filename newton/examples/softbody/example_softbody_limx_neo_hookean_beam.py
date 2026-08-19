# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody LIMX Neo-Hookean Beam
#
# A tetrahedral cantilever compares full projected-Newton steps with Armijo
# backtracking for standard logarithmic Neo-Hookean elasticity.
#
# Command: uv run -m newton.examples softbody_limx_neo_hookean_beam
#
###########################################################################

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples

GRID_DIMENSIONS = (12, 2, 2)
CELL_SIZE = 0.05
DENSITY = 1000.0
YOUNG_MODULUS = 1.0e6
POISSON_RATIO = 0.3
ANCHOR_STIFFNESS = 1.0e8


def create_cantilever_model(device: Any = None) -> newton.Model:
    """Create the undamped tetrahedral cantilever model."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.add_soft_grid(
        pos=wp.vec3(0.0, -0.05, 0.75),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=GRID_DIMENSIONS[0],
        dim_y=GRID_DIMENSIONS[1],
        dim_z=GRID_DIMENSIONS[2],
        cell_x=CELL_SIZE,
        cell_y=CELL_SIZE,
        cell_z=CELL_SIZE,
        density=DENSITY,
        k_mu=0.0,
        k_lambda=0.0,
        k_damp=0.0,
        fix_left=False,
    )
    return builder.finalize(device=device)


def create_material_constraints(
    model: newton.Model,
    rest_positions: np.ndarray,
    material: str,
) -> list[Any]:
    """Create one tetrahedral material batch for the cantilever."""
    if rest_positions.shape != (model.particle_count, 3):
        raise ValueError("rest_positions must match the cantilever particle count")
    if material != "neo_hookean":
        raise ValueError("material must be 'neo_hookean'")

    shear_modulus = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
    lame_parameter = YOUNG_MODULUS * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
    tetrahedra = model.tet_indices.numpy()
    inverse_rest_matrices = model.tet_poses.numpy()
    return [
        newton.solvers.ConstraintTetrahedronNeoHookean(
            tetrahedra.tolist(),
            [wp.mat33(*matrix.reshape(-1)) for matrix in inverse_rest_matrices],
            [shear_modulus] * model.tet_count,
            [lame_parameter] * model.tet_count,
            model.particle_count,
            model.device,
        )
    ]


class Example:
    """Simulate a fixed cantilever with logarithmic Neo-Hookean elasticity."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.grid_dimensions = GRID_DIMENSIONS
        self.frame_dt = 0.01
        self.sim_time = 0.0

        self.model = create_cantilever_model()
        self.rest_positions = self.model.particle_q.numpy()
        self.tetrahedra = self.model.tet_indices.numpy()

        minimum_x = float(np.min(self.rest_positions[:, 0]))
        maximum_x = float(np.max(self.rest_positions[:, 0]))
        self.anchor_indices = np.flatnonzero(np.isclose(self.rest_positions[:, 0], minimum_x))
        self.free_end_indices = np.flatnonzero(np.isclose(self.rest_positions[:, 0], maximum_x))
        self.initial_free_end_z = float(np.mean(self.rest_positions[self.free_end_indices, 2]))
        self.minimum_free_end_z = self.initial_free_end_z

        self.anchor_constraint = newton.solvers.ConstraintAnchor(
            self.anchor_indices.tolist(),
            [wp.vec3(*position) for position in self.rest_positions[self.anchor_indices]],
            [ANCHOR_STIFFNESS] * len(self.anchor_indices),
            self.model.particle_count,
            self.model.device,
        )
        self.material_constraints = create_material_constraints(
            self.model,
            self.rest_positions,
            "neo_hookean",
        )
        self.material_constraint = self.material_constraints[0]
        line_search = newton.solvers.SolverLIMX.LineSearch() if args.line_search else None
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            [self.anchor_constraint, *self.material_constraints],
            nonlinear_iterations=1,
            linear_iterations=256,
            velocity_damping=1.0,
            line_search=line_search,
            linear_tolerance=1.0e-6,
            nonlinear_tolerance=1.0e-5,
            record_diagnostics=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.3, -0.9, 0.95), -10.0, 90.0)

    def step(self):
        """Advance one undamped 0.01-second projected-Newton step."""
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current tetrahedral cantilever surface."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Keep the cantilever finite, positive-volume, and line-search valid."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("Neo-Hookean cantilever state must remain finite")

        minimum_determinant = self.material_constraint.minimum_determinant(self.state_0.particle_q)
        if minimum_determinant <= 0.0:
            raise AssertionError("Neo-Hookean cantilever tetrahedra must remain positive-volume")

        records = self.solver.last_step_diagnostics
        if records and records[-1].status in {
            "invalid_current",
            "invalid_candidate",
            "line_search_failed",
            "non_descent_direction",
            "nonfinite_objective",
        }:
            raise AssertionError(f"Neo-Hookean solve terminated with {records[-1].status}")

        self.minimum_free_end_z = min(
            self.minimum_free_end_z,
            float(np.mean(positions[self.free_end_indices, 2])),
        )

    def test_final(self):
        """Keep the left face anchored while the free end visibly falls."""
        positions = self.state_0.particle_q.numpy()
        np.testing.assert_allclose(
            positions[self.anchor_indices],
            self.rest_positions[self.anchor_indices],
            atol=2.0e-3,
        )
        if self.minimum_free_end_z >= self.initial_free_end_z - 2.0e-3:
            raise AssertionError("Neo-Hookean cantilever free end must fall under gravity")

    @staticmethod
    def create_parser():
        """Create the standard Newton example parser."""
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--line-search",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable Armijo backtracking for projected-Newton steps.",
        )
        parser.set_defaults(num_frames=300)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
