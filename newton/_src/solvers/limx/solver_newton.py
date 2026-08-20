# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Projected-Newton particle solver using block-CSR elasticity and PCG."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .block_csr import BlockCsrBuilder
from .linear_solver import PcgSolver
from .operator import CompositeLinearOperator, EmptyDynamicConstraintOperator


@wp.kernel
def _initialize_step(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    external_forces: wp.array[wp.vec3],
    masses: wp.array[float],
    particle_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    dt: float,
    previous_positions: wp.array[wp.vec3],
    inertia_positions: wp.array[wp.vec3],
    iterate_positions: wp.array[wp.vec3],
):
    particle = wp.tid()
    position = positions[particle]
    acceleration = gravity[wp.max(particle_world[particle], 0)] + external_forces[particle] / masses[particle]
    previous_positions[particle] = position
    inertia_positions[particle] = position + dt * velocities[particle] + dt * dt * acceleration
    iterate_positions[particle] = position


@wp.kernel
def _initialize_rhs(
    masses: wp.array[float],
    inv_dt_squared: float,
    inertia_positions: wp.array[wp.vec3],
    iterate_positions: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
):
    particle = wp.tid()
    rhs[particle] = masses[particle] * inv_dt_squared * (inertia_positions[particle] - iterate_positions[particle])


@wp.kernel
def _apply_increment(increment: wp.array[wp.vec3], iterate_positions: wp.array[wp.vec3]):
    particle = wp.tid()
    iterate_positions[particle] += increment[particle]


@wp.kernel
def _make_candidate(
    iterate_positions: wp.array[wp.vec3],
    increment: wp.array[wp.vec3],
    step_length: float,
    candidate_positions: wp.array[wp.vec3],
):
    particle = wp.tid()
    candidate_positions[particle] = iterate_positions[particle] + step_length * increment[particle]


@wp.kernel
def _accumulate_inertia_energy(
    masses: wp.array[float],
    inv_dt_squared: float,
    inertia_positions: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    energy: wp.array[float],
):
    particle = wp.tid()
    displacement = positions[particle] - inertia_positions[particle]
    wp.atomic_add(energy, 0, 0.5 * masses[particle] * inv_dt_squared * wp.dot(displacement, displacement))


@wp.kernel
def _accumulate_vector_dot(
    lhs: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
    output: wp.array[float],
):
    particle = wp.tid()
    wp.atomic_add(output, 0, wp.dot(lhs[particle], rhs[particle]))


@wp.kernel
def _finish_step(
    previous_positions: wp.array[wp.vec3],
    iterate_positions: wp.array[wp.vec3],
    inv_dt: float,
    velocity_damping: float,
    output_positions: wp.array[wp.vec3],
    output_velocities: wp.array[wp.vec3],
):
    particle = wp.tid()
    output_positions[particle] = iterate_positions[particle]
    output_velocities[particle] = (
        velocity_damping * inv_dt * (iterate_positions[particle] - previous_positions[particle])
    )


class SolverLIMX(SolverBase):
    r"""Implicit projected-Newton particle solver.

    Static constraints assemble forces and analytic positive-semidefinite
    Hessian blocks at the current Newton iterate. Their fixed topology is
    stored in a ``3 x 3`` block-CSR matrix whose values are rebuilt before
    every PCG solve. Dynamic constraints, such as future collision terms, can
    add matrix-free force, Hessian-vector, and diagonal contributions through
    ``dynamic_operator``.
    """

    @dataclass(frozen=True)
    class LineSearch:
        """Armijo backtracking parameters.

        Attributes:
            armijo_coefficient: Sufficient-decrease coefficient.
            contraction_factor: Step-length multiplier after a rejected trial.
            max_backtracks: Maximum number of step-length contractions.
        """

        armijo_coefficient: float = 1.0e-4
        contraction_factor: float = 0.5
        max_backtracks: int = 12

        def __post_init__(self):
            if not math.isfinite(self.armijo_coefficient) or not 0.0 < self.armijo_coefficient < 1.0:
                raise ValueError("armijo_coefficient must be finite and between zero and one")
            if not math.isfinite(self.contraction_factor) or not 0.0 < self.contraction_factor < 1.0:
                raise ValueError("contraction_factor must be finite and between zero and one")
            if not isinstance(self.max_backtracks, int) or isinstance(self.max_backtracks, bool):
                raise TypeError("max_backtracks must be an integer")
            if self.max_backtracks < 0:
                raise ValueError("max_backtracks must not be negative")

    @dataclass(frozen=True)
    class IterationDiagnostics:
        """Immutable diagnostics for one nonlinear iteration.

        ``status`` is one of ``accepted``, ``converged``, ``iteration_limit``,
        ``invalid_current``, ``invalid_candidate``, ``nonfinite_objective``,
        ``non_descent_direction``, or ``line_search_failed``.

        Attributes:
            iteration: Zero-based nonlinear iteration index.
            objective_before: Implicit objective before the trial step [J].
            objective_after: Implicit objective after the accepted step [J].
            gradient_norm: Euclidean objective-gradient norm [N].
            relative_gradient_norm: Gradient norm relative to the first iteration.
            step_norm: Euclidean norm of the accepted position increment [m].
            step_length: Accepted fraction of the projected-Newton direction.
            backtracks: Number of rejected larger step lengths.
            directional_derivative: Objective directional derivative [J].
            linear_iterations: Number of PCG iterations executed.
            linear_relative_residual: Final PCG residual ratio, when enabled.
            minimum_determinant: Minimum tetrahedral deformation determinant.
            status: Iteration outcome or terminal reason.
        """

        iteration: int
        objective_before: float
        objective_after: float
        gradient_norm: float
        relative_gradient_norm: float
        step_norm: float
        step_length: float
        backtracks: int
        directional_derivative: float
        linear_iterations: int
        linear_relative_residual: float | None
        minimum_determinant: float
        status: str

    def __init__(
        self,
        model: Model,
        constraints: Sequence[Any],
        nonlinear_iterations: int = 4,
        linear_iterations: int = 32,
        velocity_damping: float = 1.0,
        dynamic_operator: Any | None = None,
        line_search: LineSearch | None = None,
        linear_tolerance: float | None = None,
        nonlinear_tolerance: float | None = None,
        record_diagnostics: bool = False,
    ):
        """Create a LIMX projected-Newton particle solver.

        Args:
            model: Particle-only model containing active, positive-mass particles.
            constraints: Static constraint batches that provide current-position
                force and Hessian assembly methods.
            nonlinear_iterations: Newton position iterations per step.
            linear_iterations: Fixed PCG iterations per Newton iteration.
            velocity_damping: Per-step velocity multiplier.
            dynamic_operator: Optional matrix-free dynamic constraint operator.
            line_search: Optional Armijo backtracking configuration.
            linear_tolerance: Optional PCG relative residual tolerance.
            nonlinear_tolerance: Optional Newton relative gradient tolerance.
            record_diagnostics: Whether to record objective-aware iteration data.
        """
        super().__init__(model)
        if model.body_count > 0:
            raise ValueError("SolverLIMX is particle-only and does not accept rigid bodies")
        if model.particle_count <= 0 or model.particle_mass is None:
            raise ValueError("SolverLIMX requires at least one particle")
        masses = model.particle_mass.numpy()
        if not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError("SolverLIMX requires finite positive particle masses")
        flags = model.particle_flags.numpy()
        if np.any((flags & ParticleFlags.ACTIVE) == 0):
            raise ValueError("SolverLIMX requires active particles; use ConstraintAnchor to fix particle positions")
        if nonlinear_iterations <= 0:
            raise ValueError("nonlinear_iterations must be positive")
        if linear_iterations <= 0:
            raise ValueError("linear_iterations must be positive")
        if not np.isfinite(velocity_damping) or velocity_damping < 0.0 or velocity_damping > 1.0:
            raise ValueError("velocity_damping must be finite and between zero and one")
        if line_search is not None and not isinstance(line_search, SolverLIMX.LineSearch):
            raise TypeError("line_search must be a SolverLIMX.LineSearch or None")
        if linear_tolerance is not None and (not np.isfinite(linear_tolerance) or linear_tolerance <= 0.0):
            raise ValueError("linear_tolerance must be finite and positive")
        if nonlinear_tolerance is not None and (not np.isfinite(nonlinear_tolerance) or nonlinear_tolerance <= 0.0):
            raise ValueError("nonlinear_tolerance must be finite and positive")
        if not isinstance(record_diagnostics, bool):
            raise TypeError("record_diagnostics must be a bool")

        self.constraints = tuple(constraints)
        self.nonlinear_iterations = nonlinear_iterations
        self.linear_iterations = linear_iterations
        self.velocity_damping = float(velocity_damping)
        self.dynamic_operator = dynamic_operator if dynamic_operator is not None else EmptyDynamicConstraintOperator()
        self.line_search = line_search
        self.linear_tolerance = linear_tolerance
        self.nonlinear_tolerance = nonlinear_tolerance
        self.record_diagnostics = record_diagnostics
        self._last_step_diagnostics: tuple[SolverLIMX.IterationDiagnostics, ...] = ()
        self._objective_mode = line_search is not None or record_diagnostics
        self._needs_gradient_norm = self._objective_mode or nonlinear_tolerance is not None

        if self._objective_mode:
            for constraint in self.constraints:
                if not callable(getattr(constraint, "accumulate_energy", None)):
                    raise ValueError("Objective-aware LIMX constraints must implement accumulate_energy()")
            if not isinstance(self.dynamic_operator, EmptyDynamicConstraintOperator):
                raise ValueError("Objective-aware LIMX does not support dynamic constraint operators")

        matrix_builder = BlockCsrBuilder(model.particle_count)
        for constraint in self.constraints:
            if getattr(constraint, "particle_count", None) != model.particle_count:
                raise ValueError("Every constraint must match the model particle count")
            if getattr(constraint, "device", None) != self.device:
                raise ValueError("Every constraint must use the model device")
            constraint.append_hessian_structure(matrix_builder)
        self.static_matrix = matrix_builder.finalize(self.device)
        for constraint in self.constraints:
            constraint.bind_hessian(self.static_matrix)
        bind_dynamic_static_system = getattr(self.dynamic_operator, "bind_static_system", None)
        if bind_dynamic_static_system is not None:
            bind_dynamic_static_system(self.static_matrix.diagonal, model.particle_mass)

        self.operator = CompositeLinearOperator(
            masses=model.particle_mass,
            static_matrix=self.static_matrix,
            dynamic_operator=self.dynamic_operator,
            device=self.device,
        )
        self.linear_solver = PcgSolver(model.particle_count, self.device)

        self.previous_positions = wp.empty(model.particle_count, dtype=wp.vec3, device=self.device)
        self.inertia_positions = wp.empty_like(self.previous_positions)
        self.iterate_positions = wp.empty_like(self.previous_positions)
        self.rhs = wp.empty_like(self.previous_positions)
        self.increment = wp.zeros_like(self.previous_positions)
        self.candidate_positions = wp.empty_like(self.previous_positions) if self._objective_mode else None
        self._objective_energy = wp.zeros(1, dtype=float, device=self.device) if self._objective_mode else None
        self._objective_invalid_count = wp.zeros(1, dtype=int, device=self.device) if self._objective_mode else None
        self._scalar_result = wp.zeros(1, dtype=float, device=self.device) if self._needs_gradient_norm else None

    @property
    def last_step_diagnostics(self) -> tuple[IterationDiagnostics, ...]:
        """Return immutable nonlinear records from the most recent step."""
        return self._last_step_diagnostics

    def _evaluate_objective(self, positions: wp.array[wp.vec3], inv_dt_squared: float) -> tuple[float, int]:
        self._objective_energy.zero_()
        self._objective_invalid_count.zero_()
        wp.launch(
            _accumulate_inertia_energy,
            dim=self.model.particle_count,
            inputs=[self.model.particle_mass, inv_dt_squared, self.inertia_positions, positions],
            outputs=[self._objective_energy],
            device=self.device,
        )
        for constraint in self.constraints:
            constraint.accumulate_energy(positions, self._objective_energy, self._objective_invalid_count)
        return (
            float(self._objective_energy.numpy()[0]),
            int(self._objective_invalid_count.numpy()[0]),
        )

    def _vector_dot(self, lhs: wp.array[wp.vec3], rhs: wp.array[wp.vec3]) -> float:
        self._scalar_result.zero_()
        wp.launch(
            _accumulate_vector_dot,
            dim=self.model.particle_count,
            inputs=[lhs, rhs],
            outputs=[self._scalar_result],
            device=self.device,
        )
        return float(self._scalar_result.numpy()[0])

    def _minimum_determinant(self, positions: wp.array[wp.vec3]) -> float:
        determinants = [
            constraint.minimum_determinant(positions)
            for constraint in self.constraints
            if callable(getattr(constraint, "minimum_determinant", None))
        ]
        return min(determinants, default=math.inf)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance particles by one implicit-Euler time step.

        ``control`` and ``contacts`` are currently unused. Collision constraints
        can be supplied through the solver's matrix-free dynamic operator.

        Args:
            state_in: Input state, which remains unchanged.
            state_out: State receiving updated particle positions and velocities.
            control: Unused control input.
            contacts: Unused Newton contact data.
            dt: Simulation time step [s].
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self._last_step_diagnostics = ()

        model = self.model
        wp.launch(
            _initialize_step,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_mass,
                model.particle_world,
                model.gravity,
                dt,
            ],
            outputs=[self.previous_positions, self.inertia_positions, self.iterate_positions],
            device=self.device,
        )

        begin_dynamic_step = getattr(self.dynamic_operator, "begin_step", None)
        if begin_dynamic_step is not None:
            begin_dynamic_step(state_in.particle_q, state_in.particle_qd, dt)

        inv_dt_squared = 1.0 / (dt * dt)
        diagnostics: list[SolverLIMX.IterationDiagnostics] = []
        initial_gradient_norm: float | None = None
        iterate_objective: float | None = None
        for nonlinear_iteration in range(self.nonlinear_iterations):
            objective_before = math.nan
            if self._objective_mode:
                if iterate_objective is None:
                    objective_before, invalid_count = self._evaluate_objective(
                        self.iterate_positions,
                        inv_dt_squared,
                    )
                else:
                    # Preserve the accepted scalar so atomic reduction order cannot relax the next Armijo bound.
                    objective_before = iterate_objective
                    invalid_count = 0
                if invalid_count > 0 or not math.isfinite(objective_before):
                    if self.record_diagnostics:
                        diagnostics.append(
                            SolverLIMX.IterationDiagnostics(
                                iteration=nonlinear_iteration,
                                objective_before=objective_before,
                                objective_after=objective_before,
                                gradient_norm=math.nan,
                                relative_gradient_norm=math.nan,
                                step_norm=0.0,
                                step_length=0.0,
                                backtracks=0,
                                directional_derivative=math.nan,
                                linear_iterations=0,
                                linear_relative_residual=None,
                                minimum_determinant=self._minimum_determinant(self.iterate_positions),
                                status="invalid_current" if invalid_count > 0 else "nonfinite_objective",
                            )
                        )
                    break

            prepare_dynamic_constraints = getattr(self.dynamic_operator, "prepare", None)
            if prepare_dynamic_constraints is not None:
                prepare_dynamic_constraints(self.iterate_positions)
            self.static_matrix.clear_values()
            wp.launch(
                _initialize_rhs,
                dim=model.particle_count,
                inputs=[
                    model.particle_mass,
                    inv_dt_squared,
                    self.inertia_positions,
                    self.iterate_positions,
                ],
                outputs=[self.rhs],
                device=self.device,
            )
            for constraint in self.constraints:
                constraint.accumulate_force_and_hessian(
                    self.iterate_positions,
                    self.rhs,
                    self.static_matrix.values,
                )
            self.static_matrix.update_diagonal()
            self.dynamic_operator.accumulate_force(self.iterate_positions, self.rhs)

            gradient_norm = math.nan
            relative_gradient_norm = math.nan
            if self._needs_gradient_norm:
                gradient_norm = math.sqrt(max(self._vector_dot(self.rhs, self.rhs), 0.0))
                if initial_gradient_norm is None:
                    initial_gradient_norm = gradient_norm
                relative_gradient_norm = 0.0 if initial_gradient_norm == 0.0 else gradient_norm / initial_gradient_norm
                if self.nonlinear_tolerance is not None and relative_gradient_norm <= self.nonlinear_tolerance:
                    if self.record_diagnostics:
                        diagnostics.append(
                            SolverLIMX.IterationDiagnostics(
                                iteration=nonlinear_iteration,
                                objective_before=objective_before,
                                objective_after=objective_before,
                                gradient_norm=gradient_norm,
                                relative_gradient_norm=relative_gradient_norm,
                                step_norm=0.0,
                                step_length=0.0,
                                backtracks=0,
                                directional_derivative=0.0,
                                linear_iterations=0,
                                linear_relative_residual=None,
                                minimum_determinant=self._minimum_determinant(self.iterate_positions),
                                status="converged",
                            )
                        )
                    break

            self.operator.prepare(self.iterate_positions, dt)
            if self.linear_tolerance is None:
                linear_iterations = self.linear_solver.solve(
                    self.operator,
                    self.rhs,
                    self.increment,
                    iterations=self.linear_iterations,
                    zero_initial_guess=nonlinear_iteration > 0,
                )
            else:
                linear_iterations = self.linear_solver.solve(
                    self.operator,
                    self.rhs,
                    self.increment,
                    iterations=self.linear_iterations,
                    zero_initial_guess=nonlinear_iteration > 0,
                    relative_tolerance=self.linear_tolerance,
                )

            if not self._objective_mode:
                wp.launch(
                    _apply_increment,
                    dim=model.particle_count,
                    inputs=[self.increment],
                    outputs=[self.iterate_positions],
                    device=self.device,
                )
                continue

            increment_norm = math.sqrt(max(self._vector_dot(self.increment, self.increment), 0.0))
            directional_derivative = -self._vector_dot(self.rhs, self.increment)
            linear_relative_residual = (
                self.linear_solver.last_relative_residual if self.linear_tolerance is not None else None
            )

            if self.line_search is not None:
                if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
                    if self.record_diagnostics:
                        diagnostics.append(
                            SolverLIMX.IterationDiagnostics(
                                iteration=nonlinear_iteration,
                                objective_before=objective_before,
                                objective_after=objective_before,
                                gradient_norm=gradient_norm,
                                relative_gradient_norm=relative_gradient_norm,
                                step_norm=0.0,
                                step_length=0.0,
                                backtracks=0,
                                directional_derivative=directional_derivative,
                                linear_iterations=linear_iterations,
                                linear_relative_residual=linear_relative_residual,
                                minimum_determinant=self._minimum_determinant(self.iterate_positions),
                                status="non_descent_direction",
                            )
                        )
                    break

                step_length = 1.0
                accepted = False
                objective_after = math.nan
                backtracks = 0
                for candidate_index in range(self.line_search.max_backtracks + 1):
                    backtracks = candidate_index
                    wp.launch(
                        _make_candidate,
                        dim=model.particle_count,
                        inputs=[self.iterate_positions, self.increment, step_length],
                        outputs=[self.candidate_positions],
                        device=self.device,
                    )
                    objective_after, invalid_count = self._evaluate_objective(
                        self.candidate_positions,
                        inv_dt_squared,
                    )
                    if (
                        invalid_count == 0
                        and math.isfinite(objective_after)
                        and objective_after
                        <= objective_before + self.line_search.armijo_coefficient * step_length * directional_derivative
                    ):
                        wp.copy(self.iterate_positions, self.candidate_positions)
                        iterate_objective = objective_after
                        accepted = True
                        break
                    step_length *= self.line_search.contraction_factor

                if not accepted:
                    if self.record_diagnostics:
                        diagnostics.append(
                            SolverLIMX.IterationDiagnostics(
                                iteration=nonlinear_iteration,
                                objective_before=objective_before,
                                objective_after=objective_before,
                                gradient_norm=gradient_norm,
                                relative_gradient_norm=relative_gradient_norm,
                                step_norm=0.0,
                                step_length=0.0,
                                backtracks=backtracks,
                                directional_derivative=directional_derivative,
                                linear_iterations=linear_iterations,
                                linear_relative_residual=linear_relative_residual,
                                minimum_determinant=self._minimum_determinant(self.iterate_positions),
                                status="line_search_failed",
                            )
                        )
                    break

                if self.record_diagnostics:
                    diagnostics.append(
                        SolverLIMX.IterationDiagnostics(
                            iteration=nonlinear_iteration,
                            objective_before=objective_before,
                            objective_after=objective_after,
                            gradient_norm=gradient_norm,
                            relative_gradient_norm=relative_gradient_norm,
                            step_norm=step_length * increment_norm,
                            step_length=step_length,
                            backtracks=backtracks,
                            directional_derivative=directional_derivative,
                            linear_iterations=linear_iterations,
                            linear_relative_residual=linear_relative_residual,
                            minimum_determinant=self._minimum_determinant(self.iterate_positions),
                            status=(
                                "iteration_limit"
                                if nonlinear_iteration + 1 == self.nonlinear_iterations
                                else "accepted"
                            ),
                        )
                    )
            else:
                wp.launch(
                    _make_candidate,
                    dim=model.particle_count,
                    inputs=[self.iterate_positions, self.increment, 1.0],
                    outputs=[self.candidate_positions],
                    device=self.device,
                )
                objective_after, invalid_count = self._evaluate_objective(
                    self.candidate_positions,
                    inv_dt_squared,
                )
                wp.copy(self.iterate_positions, self.candidate_positions)
                failed = invalid_count > 0 or not math.isfinite(objective_after)
                if not failed:
                    iterate_objective = objective_after
                if self.record_diagnostics:
                    diagnostics.append(
                        SolverLIMX.IterationDiagnostics(
                            iteration=nonlinear_iteration,
                            objective_before=objective_before,
                            objective_after=objective_after,
                            gradient_norm=gradient_norm,
                            relative_gradient_norm=relative_gradient_norm,
                            step_norm=increment_norm,
                            step_length=1.0,
                            backtracks=0,
                            directional_derivative=directional_derivative,
                            linear_iterations=linear_iterations,
                            linear_relative_residual=linear_relative_residual,
                            minimum_determinant=self._minimum_determinant(self.iterate_positions),
                            status=(
                                "invalid_candidate"
                                if invalid_count > 0
                                else "nonfinite_objective"
                                if not math.isfinite(objective_after)
                                else "iteration_limit"
                                if nonlinear_iteration + 1 == self.nonlinear_iterations
                                else "accepted"
                            ),
                        )
                    )
                if failed:
                    break

        self._last_step_diagnostics = tuple(diagnostics)

        wp.launch(
            _finish_step,
            dim=model.particle_count,
            inputs=[self.previous_positions, self.iterate_positions, 1.0 / dt, self.velocity_damping],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=self.device,
        )
