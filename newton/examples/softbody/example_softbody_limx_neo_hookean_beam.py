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
import csv
import math
from dataclasses import asdict
from pathlib import Path
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
    if material not in {"neo_hookean", "quadratic"}:
        raise ValueError("material must be 'neo_hookean' or 'quadratic'")

    shear_modulus = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
    lame_parameter = YOUNG_MODULUS * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
    tetrahedra = model.tet_indices.numpy()
    inverse_rest_matrices = model.tet_poses.numpy()
    constraint_type = (
        newton.solvers.ConstraintTetrahedronNeoHookean
        if material == "neo_hookean"
        else newton.solvers.ConstraintTetrahedronLinearElastic
    )
    return [
        constraint_type(
            tetrahedra.tolist(),
            [wp.mat33(*matrix.reshape(-1)) for matrix in inverse_rest_matrices],
            [shear_modulus] * model.tet_count,
            [lame_parameter] * model.tet_count,
            model.particle_count,
            model.device,
        )
    ]


def _create_study_solver(
    material: str,
    device: Any,
    *,
    line_search: bool,
    nonlinear_iterations: int,
    linear_iterations: int = 256,
    linear_tolerance: float = 1.0e-6,
    nonlinear_tolerance: float = 1.0e-5,
):
    model = create_cantilever_model(device)
    rest_positions = model.particle_q.numpy()
    minimum_x = float(np.min(rest_positions[:, 0]))
    anchor_indices = np.flatnonzero(np.isclose(rest_positions[:, 0], minimum_x))
    anchor = newton.solvers.ConstraintAnchor(
        anchor_indices.tolist(),
        [wp.vec3(*position) for position in rest_positions[anchor_indices]],
        [ANCHOR_STIFFNESS] * len(anchor_indices),
        model.particle_count,
        model.device,
    )
    constraints = [anchor, *create_material_constraints(model, rest_positions, material)]
    solver = newton.solvers.SolverLIMX(
        model,
        constraints,
        nonlinear_iterations=nonlinear_iterations,
        linear_iterations=linear_iterations,
        velocity_damping=1.0,
        line_search=newton.solvers.SolverLIMX.LineSearch() if line_search else None,
        linear_tolerance=linear_tolerance,
        nonlinear_tolerance=nonlinear_tolerance,
        record_diagnostics=True,
    )
    return model, solver


def _create_checkpoint(device: Any, checkpoint_time: float) -> tuple[np.ndarray, np.ndarray]:
    checkpoint_steps = round(checkpoint_time / 0.01)
    if checkpoint_steps < 0 or not math.isclose(checkpoint_steps * 0.01, checkpoint_time, abs_tol=1.0e-12):
        raise ValueError("checkpoint_time must be a nonnegative multiple of 0.01 seconds")

    model, solver = _create_study_solver(
        "neo_hookean",
        device,
        line_search=True,
        nonlinear_iterations=1,
    )
    state_0 = model.state()
    state_1 = model.state()
    for _ in range(checkpoint_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, 0.01)
        state_0, state_1 = state_1, state_0
    return state_0.particle_q.numpy().copy(), state_0.particle_qd.numpy().copy()


def _run_study_solve(
    material: str,
    device: Any,
    positions: np.ndarray,
    velocities: np.ndarray,
    time_step: float,
    *,
    line_search: bool,
    nonlinear_iterations: int,
    linear_iterations: int = 256,
    linear_tolerance: float = 1.0e-6,
    nonlinear_tolerance: float = 1.0e-5,
):
    model, solver = _create_study_solver(
        material,
        device,
        line_search=line_search,
        nonlinear_iterations=nonlinear_iterations,
        linear_iterations=linear_iterations,
        linear_tolerance=linear_tolerance,
        nonlinear_tolerance=nonlinear_tolerance,
    )
    state_in = model.state()
    state_out = model.state()
    state_in.particle_q.assign(positions)
    state_in.particle_qd.assign(velocities)
    state_in.clear_forces()
    solver.step(state_in, state_out, None, None, time_step)
    return solver.last_step_diagnostics


def _minimum_finite_objective(records) -> float:
    finite_objectives = []
    for record in records:
        if record.status != "invalid_current" and math.isfinite(record.objective_before):
            finite_objectives.append(record.objective_before)
        if record.status not in {"invalid_current", "invalid_candidate", "nonfinite_objective"} and math.isfinite(
            record.objective_after
        ):
            finite_objectives.append(record.objective_after)
    if not finite_objectives:
        raise RuntimeError("Convergence solve produced no finite objective")
    return min(finite_objectives)


def _write_convergence_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    field_names = [
        "time_step",
        "method",
        "iteration",
        "objective_before",
        "objective_after",
        "gradient_norm",
        "relative_gradient_norm",
        "step_norm",
        "step_length",
        "backtracks",
        "directional_derivative",
        "linear_iterations",
        "linear_relative_residual",
        "minimum_determinant",
        "status",
        "objective_baseline",
        "baseline_source",
        "tight_reference_objective",
        "tight_reference_status",
        "tight_reference_relative_gradient_norm",
        "objective_gap",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in field_names})


def _write_convergence_plot(
    rows: list[dict[str, float | int | str]],
    time_steps: tuple[float, ...],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    methods = (
        ("quadratic", "Quadratic", "#4c78a8", "-"),
        ("neo_hookean_full", "Neo-Hookean full", "#f58518", "--"),
        ("neo_hookean_armijo", "Neo-Hookean Armijo", "#54a24b", "-"),
    )
    figure, axes = plt.subplots(2, len(time_steps), figsize=(4.2 * len(time_steps), 6.4), squeeze=False)
    for column, time_step in enumerate(time_steps):
        for method, label, color, line_style in methods:
            method_rows = [
                row for row in rows if row["method"] == method and math.isclose(float(row["time_step"]), time_step)
            ]
            if not method_rows:
                continue
            iterations = [int(row["iteration"]) for row in method_rows]
            relative_gradients = [float(row["relative_gradient_norm"]) for row in method_rows]
            objective_gaps = [
                max(float(row["objective_gap"]), 1.0e-12) if math.isfinite(float(row["objective_gap"])) else math.nan
                for row in method_rows
            ]
            axes[0, column].plot(
                iterations,
                relative_gradients,
                color=color,
                linestyle=line_style,
                marker="o",
                label=label,
            )
            axes[1, column].plot(
                iterations,
                objective_gaps,
                color=color,
                linestyle=line_style,
                marker="o",
                label=label,
            )
            terminal = method_rows[-1]
            if terminal["status"] != "accepted":
                terminal_values = (relative_gradients[-1], objective_gaps[-1])
                for row, value in enumerate(terminal_values):
                    if math.isfinite(value):
                        axes[row, column].scatter(
                            [iterations[-1]],
                            [value],
                            color=color,
                            marker="x",
                            s=45,
                            zorder=5,
                        )
                if math.isfinite(relative_gradients[-1]):
                    axes[0, column].annotate(
                        str(terminal["status"]),
                        (iterations[-1], relative_gradients[-1]),
                        fontsize=7,
                        color=color,
                    )

        axes[0, column].set_title(f"dt = {time_step:.2f} s")
        for row in range(2):
            axes[row, column].set_yscale("log")
            axes[row, column].set_xlabel("Newton iteration")
            axes[row, column].grid(True, which="both", alpha=0.3)

    axes[0, 0].set_ylabel("Relative gradient norm")
    axes[1, 0].set_ylabel("Normalized gap to best observed")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_convergence_study(
    device: str,
    output_directory: Path,
    *,
    checkpoint_time: float = 0.30,
    time_steps: tuple[float, ...] = (0.01, 0.03, 0.05),
    max_newton_iterations: int = 20,
) -> list[dict[str, float | int | str]]:
    """Run deterministic single-step convergence comparisons and write CSV/PNG output."""
    if max_newton_iterations <= 0:
        raise ValueError("max_newton_iterations must be positive")
    if not time_steps or any(not math.isfinite(dt) or dt <= 0.0 for dt in time_steps):
        raise ValueError("time_steps must contain finite positive values")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    positions, velocities = _create_checkpoint(device, checkpoint_time)
    method_configs = (
        ("quadratic", "quadratic", False),
        ("neo_hookean_full", "neo_hookean", False),
        ("neo_hookean_armijo", "neo_hookean", True),
    )

    tight_references = {}
    for time_step in time_steps:
        for material in ("quadratic", "neo_hookean"):
            reference_records = _run_study_solve(
                material,
                device,
                positions,
                velocities,
                time_step,
                line_search=True,
                nonlinear_iterations=max(20, max_newton_iterations),
                linear_iterations=512,
                linear_tolerance=1.0e-7,
                nonlinear_tolerance=1.0e-6,
            )
            _minimum_finite_objective(reference_records)
            tight_references[(time_step, material)] = reference_records

    run_records = {}
    for time_step in time_steps:
        for method, material, line_search in method_configs:
            run_records[(time_step, method)] = _run_study_solve(
                material,
                device,
                positions,
                velocities,
                time_step,
                line_search=line_search,
                nonlinear_iterations=max_newton_iterations,
            )

    baselines = {}
    for time_step in time_steps:
        for material in ("quadratic", "neo_hookean"):
            candidates = [
                (
                    _minimum_finite_objective(tight_references[(time_step, material)]),
                    "tight_reference",
                )
            ]
            for method, candidate_material, _ in method_configs:
                if candidate_material == material:
                    candidates.append(
                        (
                            _minimum_finite_objective(run_records[(time_step, method)]),
                            method,
                        )
                    )
            baselines[(time_step, material)] = min(candidates, key=lambda candidate: candidate[0])

    rows: list[dict[str, float | int | str]] = []
    for time_step in time_steps:
        for method, material, _ in method_configs:
            records = run_records[(time_step, method)]
            baseline, baseline_source = baselines[(time_step, material)]
            tight_reference = tight_references[(time_step, material)]
            tight_reference_terminal = tight_reference[-1]
            tight_reference_objective = _minimum_finite_objective(tight_reference)
            initial_objective = records[0].objective_before
            normalization = max(abs(initial_objective - baseline), 1.0e-12)
            for record in records:
                row = {"time_step": time_step, "method": method, **asdict(record)}
                row["objective_baseline"] = baseline
                row["baseline_source"] = baseline_source
                row["tight_reference_objective"] = tight_reference_objective
                row["tight_reference_status"] = tight_reference_terminal.status
                row["tight_reference_relative_gradient_norm"] = tight_reference_terminal.relative_gradient_norm
                row["objective_gap"] = (
                    max((record.objective_after - baseline) / normalization, 0.0)
                    if math.isfinite(record.objective_after)
                    else math.inf
                )
                rows.append(row)

    _write_convergence_csv(rows, output_directory / "limx_neo_hookean_convergence.csv")
    _write_convergence_plot(rows, time_steps, output_directory / "limx_neo_hookean_convergence.png")
    return rows


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
        parser.add_argument(
            "--convergence-study",
            action="store_true",
            help="Write the deterministic convergence CSV and PNG instead of the visual rollout.",
        )
        parser.add_argument(
            "--output-directory",
            type=Path,
            default=Path("."),
            help="Directory for convergence-study output files.",
        )
        parser.set_defaults(num_frames=300)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.convergence_study:
        selected_device = wp.get_device(args.device) if args.device else wp.get_device()
        run_convergence_study(str(selected_device), args.output_directory)
    else:
        newton.examples.run(Example(viewer, args), args)
