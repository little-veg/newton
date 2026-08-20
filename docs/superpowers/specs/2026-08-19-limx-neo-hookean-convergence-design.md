# LIMX Neo-Hookean Convergence Study Design

## Goal

Add a controlled LIMX experiment that measures how a genuinely nonlinear
tetrahedral elastic energy affects projected-Newton convergence. A horizontal
tetrahedral cantilever is fixed at its left face and falls under gravity. The
study compares:

1. a quadratic small-strain elastic baseline;
2. standard compressible Neo-Hookean elasticity with full Newton steps; and
3. the same Neo-Hookean formulation with Armijo backtracking.

The Neo-Hookean model is the standard logarithmic formulation containing
`log(J)` and `F^-T`. It is deliberately not Newton's existing polynomial
"stable Neo-Hookean" VBD formulation. The experiment uses the production LIMX
block-CSR assembly and PCG path rather than a separate dense reference solver.

## Questions Answered

The experiment should make the following questions observable rather than
assuming their answers:

- Does the quadratic objective reach the linear-solve accuracy floor in one
  Newton iteration?
- How does Neo-Hookean relative gradient convergence change as the time step
  grows from `0.01 s` to `0.05 s`?
- Does accepting every full step increase the objective, approach element
  inversion, oscillate, or fail for the selected state?
- When does Armijo backtracking shorten a step, and does the accepted sequence
  retain positive tetrahedron determinants and monotonically decrease the
  implicit objective?
- Are apparent nonlinear convergence limits actually caused by PCG error?

No test will encode the research hypothesis that Neo-Hookean elasticity must
converge more slowly or that line search must use fewer Newton iterations.

## Scope

This milestone includes:

- public tetrahedral quadratic-linear-elastic and standard compressible
  Neo-Hookean constraints for `SolverLIMX`;
- complete analytical `9 x 9` deformation-gradient Hessians;
- per-element PSD projection for the Neo-Hookean Hessian;
- an energy-accumulation protocol sufficient for inertia, anchors, and the two
  experiment materials;
- optional Armijo backtracking in `SolverLIMX`, disabled by default;
- optional nonlinear and linear convergence diagnostics;
- a tetrahedral cantilever example with a line-search switch;
- a deterministic single-step convergence study and reproducible PNG/CSV
  output;
- focused CUDA tests, public exports, API generation, README registration,
  example image, and an `[Unreleased]` changelog entry.

This milestone excludes:

- the polynomial stable Neo-Hookean material used by VBD;
- strict Newton with an indefinite global Hessian;
- MINRES, direct sparse factorization, or a second linear-solver backend;
- contact, collision, self-collision, friction, or material damping;
- line search for matrix-free dynamic constraints;
- exact elimination of fixed degrees of freedom;
- inversion recovery after accepting an invalid full step;
- claims that one tested scene establishes a general convergence theorem.

## Scene

The dynamic example uses a regular tetrahedral beam:

- dimensions: `0.60 x 0.10 x 0.10 m`;
- resolution: `12 x 2 x 2` hexahedral cells split into tetrahedra;
- density: `1000 kg/m^3`;
- Young's modulus: `1.0 MPa`;
- Poisson ratio: `0.3`;
- gravity: `(0, 0, -9.81) m/s^2`;
- initial condition: horizontal and at rest;
- fixed region: every particle in the leftmost position layer;
- anchor stiffness: `1.0e8 N/m`;
- default visual time step: `0.01 s`;
- no collision, damping, or velocity multiplier below one.

The material constructor receives Lamé parameters converted from the common
Young's modulus and Poisson ratio. Both material variants use the same Lamé
parameters, rest mesh, mass, anchor targets, gravity, and input state.

The anchors remain quadratic penalties because that is the existing LIMX
fixed-particle mechanism. Their energy is included in the objective. Exact
fixed-DOF elimination would change the block system and is intentionally a
separate problem.

## Deformation Kinematics

For current tetrahedron vertices `x0`, `x1`, `x2`, and `x3`, define

```text
Ds = [x1 - x0, x2 - x0, x3 - x0]
F  = Ds * Dm_inverse
J  = det(F)
V0 = 1 / (6 * det(Dm_inverse)).
```

`Dm_inverse` and `V0` are validated once during construction. Every rest
tetrahedron must have positive volume. Deformation-gradient coordinates use
the existing LIMX column-major convention
`index(spatial, material) = 3 * material + spatial`.

The four constant material gradients are the rows of `Dm_inverse` for local
vertices one through three and the negative row sum for local vertex zero.
They map material derivatives to four forces and all sixteen ordered `3 x 3`
particle-pair Hessian blocks, matching `ConstraintTetrahedronARAP`.

## Quadratic Baseline

Let

```text
epsilon = sym(F - I).
```

The quadratic baseline energy is

```text
Psi_Q = V0 * (mu * epsilon:epsilon
              + 0.5 * lambda * trace(epsilon)^2).
```

Its first Piola derivative and Hessian action are

```text
P_Q         = V0 * (2 * mu * epsilon
                    + lambda * trace(epsilon) * I)
dP_Q[dF]    = V0 * (mu * (dF + transpose(dF))
                    + lambda * trace(dF) * I).
```

The Hessian is constant and PSD. Together with quadratic inertia and anchors,
the complete implicit objective is quadratic. An adequately converged PCG
solve should therefore reduce its gradient to the linear-solve accuracy floor
in one Newton iteration. The energy is not objective under finite rigid
rotation and is used only as a numerical baseline, not as the preferred
large-deformation physical material.

The public constraint is named
`newton.solvers.ConstraintTetrahedronLinearElastic`.

## Standard Compressible Neo-Hookean Energy

For `J > 0`, use

```text
Psi_NH = V0 * (
    0.5 * mu * (trace(transpose(F) * F) - 3)
    - mu * log(J)
    + 0.5 * lambda * log(J)^2
).
```

The first Piola derivative is

```text
q    = lambda * log(J) - mu
P_NH = V0 * (mu * F + q * inverse(transpose(F))).
```

For `A = inverse(transpose(F))`, its exact Hessian action is

```text
dP_NH[dF] = V0 * (
    mu * dF
    + lambda * (A:dF) * A
    - q * A * transpose(dF) * A
).
```

Equivalently, the material Hessian entries are

```text
H_(i,a),(j,b) = V0 * (
    mu * delta(i,j) * delta(a,b)
    + lambda * A(i,a) * A(j,b)
    - q * A(i,b) * A(j,a)
).
```

Energy, gradient, and unprojected Hessian are undefined for `J <= 0`. Runtime
kernels must test the determinant before evaluating `log(J)` or `F^-T` and
mark invalid candidates without producing a NaN that can contaminate another
element's reduction.

The public constraint is named
`newton.solvers.ConstraintTetrahedronNeoHookean`.

## Projected Hessian

The experiment remains a projected-Newton method compatible with LIMX PCG.
For every valid tetrahedron:

1. assemble the complete symmetric `9 x 9` Neo-Hookean material Hessian;
2. compute all eigenpairs with the existing Warp symmetric QR eigensolver;
3. replace every negative eigenvalue with zero;
4. reconstruct the PSD material Hessian; and
5. map it to all sixteen global particle-pair blocks.

The force always comes from the exact Neo-Hookean gradient. Only the Hessian
used to compute the direction is projected. The implicit inertia and anchor
terms make the complete system positive definite even when an elastic mode is
clamped to zero.

The implementation and documentation must call this projected Newton, not
strict Newton. The unprojected Hessian remains available to focused derivative
tests but is not passed to PCG.

## Implicit Objective and Energy Protocol

For a time step of length `dt`, LIMX minimizes

```text
Phi(x) = sum_i 0.5 * mass_i / dt^2 * ||x_i - y_i||^2
         + sum_constraints Psi_constraint(x),
```

where `y` is the existing inertia position. Its negative gradient is exactly
the current LIMX right-hand side:

```text
-grad(Phi) = M / dt^2 * (y - x) + physical_elastic_force.
```

Static constraints participating in an objective-aware solve implement

```python
accumulate_energy(positions, scalar_output, invalid_output)
```

where the scalar output accumulates joules and the invalid output records an
undefined domain such as `J <= 0`. This milestone implements the protocol for
anchors and the two new tetrahedral constraints. Solver construction raises a
clear error if line search or objective diagnostics are requested with a
constraint that lacks energy evaluation.

Matrix-free dynamic constraints are unsupported by this first line-search
path. The solver rejects line search when a non-empty dynamic operator is
present instead of silently omitting part of the objective.

## Armijo Backtracking

The existing full-step behavior remains the default. With line search
enabled, let `p` be the projected-Newton direction, `g = grad(Phi(x))`, and
begin with `alpha = 1`. Accept the first valid candidate satisfying

```text
Phi(x + alpha * p) <= Phi(x) + c1 * alpha * dot(g, p),
```

using:

- `c1 = 1.0e-4`;
- contraction factor `0.5`;
- at most 12 backtracks.

A candidate with any non-finite state, `J <= 0`, or non-finite objective is
invalid and cannot be accepted. If no candidate satisfies Armijo within the
budget, keep the current iterate, record line-search failure, and terminate
that nonlinear solve.

If numerical PCG error produces `dot(g, p) >= 0`, do not pretend that Armijo
can certify the direction. Record a non-descent-direction failure and stop.

For the no-line-search experiment, `alpha` is always one. Diagnostics evaluate
the resulting candidate. If it is invalid, the full step remains classified as
failed and its convergence curve terminates; it is not repaired by a hidden
clamp or fallback backtracking pass.

## Solver API and Compatibility

`SolverLIMX` gains a nested, self-contained line-search configuration with the
approved defaults, conceptually:

```python
SolverLIMX.LineSearch(
    armijo_coefficient=1.0e-4,
    contraction_factor=0.5,
    max_backtracks=12,
)
```

The constructor accepts the configuration or `None`, plus optional relative
linear/nonlinear tolerances and a diagnostics switch. Existing callers that
pass none of these options retain fixed PCG iteration counts, fixed Newton
iteration counts, full steps, and no new device-to-host synchronization.

When diagnostics are enabled, `SolverLIMX` exposes immutable nested
per-iteration records for the most recent step. Each record contains:

- nonlinear iteration index;
- objective before/after the accepted step;
- absolute and relative gradient norm;
- increment norm;
- accepted step length;
- backtracking count;
- PCG iterations and final relative residual;
- minimum tetrahedron determinant;
- convergence or failure status.

The benchmark uses a PCG relative residual tolerance of `1.0e-6` with a cap of
256 iterations. Newton uses a relative gradient tolerance of `1.0e-5` with a
cap of 20 iterations. The default production behavior remains cap-only unless
a tolerance is explicitly supplied.

The PCG implementation must compute its tolerance relative to the initial
linear residual and report the final relative residual. This diagnostic mode
may synchronize at residual checks; the existing fixed-iteration path remains
allocation-free and synchronization-free after construction.

## Convergence Benchmark

The benchmark avoids comparing different incoming states:

1. Build the cantilever at rest.
2. Produce a deterministic common checkpoint by running logarithmic
   Neo-Hookean elasticity with Armijo line search at `dt = 0.01 s` until
   physical time `t = 0.30 s`.
3. Copy the checkpoint positions and velocities.
4. From that exact state, solve one step at each of `dt = 0.01`, `0.03`, and
   `0.05 s` with:
   - quadratic linear elasticity;
   - logarithmic Neo-Hookean elasticity with full steps;
   - logarithmic Neo-Hookean elasticity with Armijo backtracking.

For each material and time step, a tighter-tolerance Armijo run participates in
the objective-baseline search. The `1.0e8 N/m` anchors and fp32 particle state
can reach an objective-resolution floor before the requested gradient
tolerance, so the plot does not claim that this run is a converged optimum.
Instead, the lowest finite objective observed across the tight run and the
measured runs defines a best-observed baseline. The CSV records the baseline
source plus the tight run's terminal status and relative gradient norm so a
poor reference cannot be hidden by the normalized plot. Objective gaps are
normalized within each objective; raw energy magnitudes are never compared
between the quadratic and Neo-Hookean materials.

The quadratic checkpoint is intentionally the same finite-deformation state
as the Neo-Hookean runs. Its non-objective large-rotation behavior is not
interpreted physically; it verifies the numerical expectation for a quadratic
objective.

## Output

The convergence-study command writes:

- `limx_neo_hookean_convergence.png`;
- `limx_neo_hookean_convergence.csv`.

The PNG is a `2 x 3` plot. Columns correspond to the three time steps. The top
row plots relative gradient norm on a logarithmic scale; the bottom row plots
the normalized gap to the best observed objective on a logarithmic scale.
Curves identify the quadratic baseline, Neo-Hookean full step, and Neo-Hookean
Armijo. A terminal marker and annotation indicate inversion, non-finite state,
non-descent direction, line-search exhaustion, or iteration limit.

The CSV uses one row per recorded Newton iteration and includes every solver
diagnostic plus material, line-search mode, time step, run status,
best-observed objective, baseline source, and tight-reference terminal
metadata. It is written with the Python standard library. Matplotlib is
imported locally from Newton's existing `examples` extra; no dependency is
added.

The dynamic example provides an explicit line-search on/off option and renders
the same beam without contact. It also implements `test_post_step()` and
`test_final()`.

## Validation

All new tests use `unittest`, and every test method has the required imperative
docstring. Focused CUDA tests cover:

1. logarithmic Neo-Hookean rest energy and force;
2. energy-gradient agreement by centered finite differences for a positive,
   nontrivially deformed tetrahedron;
3. the complete analytical unprojected `9 x 9` Hessian against finite
   differences of the gradient;
4. the projected Hessian against NumPy `eigh`, including clamping and
   reconstruction;
5. assembled `12 x 12` symmetry, PSD behavior, translation nullspace, and
   force balance;
6. rejection of non-positive rest volumes, invalid material parameters, and
   `J <= 0` trial states;
7. the quadratic material's constant Hessian and one-Newton-step convergence
   to the PCG accuracy floor;
8. a constructed distorted-tetrahedron case in which a full step is invalid
   or increases the objective while Armijo backtracks, preserves `J > 0`, and
   satisfies sufficient decrease;
9. complete diagnostics and consistent PCG/Newton stopping reasons;
10. cantilever finiteness, positive volume, anchored left face, and visible
    free-end descent.

Armijo tests require accepted objective values to be monotonically
non-increasing. They do not require fewer Newton iterations than full steps.
The experimental full-step curve is allowed to converge, oscillate, hit the
iteration cap, or fail, and the observed result is reported rather than
encoded in advance.

Before implementation is declared complete:

- demonstrate the relevant regression tests fail before their implementation
  and pass afterward;
- run focused CUDA constraint, solver, and example tests;
- run the three-time-step benchmark and inspect the CSV for finite,
  self-consistent diagnostics;
- inspect the generated PNG for legible curves and correct failure markers;
- run `docs/generate_api.py` for the two new public constraints and nested
  solver types;
- run `uvx pre-commit run -a` before committing implementation.

## Documentation and Changelog

Register the example in `README.md` with its `python -m newton.examples`
command and a `320 x 320` screenshot. Public docstrings use SI units and only
public cross-references. Examples import public symbols through
`newton.solvers`, never `newton._src`.

Because this introduces public constraints and optional solver behavior, add
an imperative present-tense entry at a random position in the appropriate
`[Unreleased]` changelog category. No public API is removed or renamed.
