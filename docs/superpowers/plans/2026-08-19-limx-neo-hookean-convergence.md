# LIMX Neo-Hookean Convergence Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add logarithmic compressible Neo-Hookean tetrahedra, optional Armijo projected-Newton steps, and a reproducible cantilever convergence study to LIMX.

**Architecture:** Two public static tetrahedral constraints share private deformation-gradient, validation, mapping, and PSD-projection helpers while retaining separate material kernels. `SolverLIMX` gains an opt-in objective-aware path that composes inertia and constraint energies, uses relative-tolerance PCG, records immutable iteration diagnostics, and performs Armijo backtracking without changing the existing default full-step path. One public example owns both the visual cantilever and the deterministic single-step CSV/PNG study.

**Tech Stack:** Python 3.12, Warp CUDA kernels, NumPy references, LIMX `3 x 3` block-CSR, block-Jacobi PCG, stdlib `csv`, locally imported Matplotlib, and `unittest` through Newton's test runner.

## Global Constraints

- Work in the canonical `/home/limx/github/newton` checkout on `dev`; do not create a worktree.
- Preserve the existing dirty files `docs/images/examples/example_cloth_limx_three_tshirts_box.jpg`, `lessons.md`, and `solver_convergence.png`; never stage them with this feature unless a later explicit decision says otherwise.
- Use standard compressible logarithmic Neo-Hookean energy with `log(J)` and `F^-T`; do not substitute VBD's polynomial stable Neo-Hookean model.
- Call the method projected Newton because the exact material Hessian is projected PSD before PCG.
- Keep line search, nonlinear tolerance, linear relative tolerance, and diagnostics opt-in so existing LIMX callers retain fixed iteration counts and no new host synchronization.
- Reject `J <= 0` before evaluating `log(J)` or `F^-T`; never repair a no-line-search full step with hidden backtracking.
- Line search supports only energy-aware static constraints and the empty dynamic operator in this milestone.
- Use `dt = 0.01, 0.03, 0.05 s`, Armijo `c1 = 1.0e-4`, contraction `0.5`, at most 12 backtracks, PCG relative tolerance `1.0e-6` with a 256-iteration cap, and Newton relative gradient tolerance `1.0e-5` with a 20-iteration cap in the study.
- Use a `12 x 2 x 2`-cell, `0.60 x 0.10 x 0.10 m` cantilever with density `1000 kg/m^3`, `E = 1.0 MPa`, `nu = 0.3`, left-layer anchor stiffness `1.0e8 N/m`, gravity only, and no contact or damping.
- Use public imports in examples and documentation; `newton._src` imports are permitted only in tests.
- Add no dependency: Matplotlib already belongs to Newton's `examples` extra and must be imported inside plotting code.
- Use `unittest`, give every test a triple-double-quoted imperative docstring, validate on CUDA by default, and never call `wp.synchronize*()` immediately before `.numpy()`.
- Follow PEP 604 unions, bracket Warp array annotations, Google-style public docstrings with SI units, and prefix-first public naming.
- Run `docs/generate_api.py` after public exports, add a random-position `[Unreleased] / Added` changelog entry, register the example and `320 x 320` image in `README.md`, and run `uvx pre-commit run -a` before the final implementation commit.

---

## File Structure

### New files

- `newton/_src/solvers/limx/constraints/tetrahedron_elastic_common.py`: private tetrahedron validation, kinematics, material-gradient mapping, `9 x 9` PSD projection, CSR binding base, and minimum-`J` reduction shared only by the two new materials.
- `newton/_src/solvers/limx/constraints/tetrahedron_linear_elastic.py`: constant-Hessian quadratic small-strain constraint and energy kernel.
- `newton/_src/solvers/limx/constraints/tetrahedron_neo_hookean.py`: logarithmic energy, exact gradient/Hessian, invalid-domain detection, and projected assembly.
- `newton/tests/test_constraint_tetrahedron_elastic.py`: single-tetrahedron construction, derivative, PSD, assembly, invalid-domain, and public-export tests.
- `newton/examples/softbody/example_softbody_limx_neo_hookean_beam.py`: cantilever builder, visual rollout, checkpoint generation, convergence runs, CSV writer, and plotter.
- `newton/tests/test_example_softbody_limx_neo_hookean_beam.py`: focused CUDA example and reduced convergence-output tests.
- `docs/images/examples/example_softbody_limx_neo_hookean_beam.jpg`: `320 x 320` example screenshot.

### Modified files

- `newton/_src/solvers/limx/constraints/anchor.py`: add quadratic anchor energy accumulation.
- `newton/_src/solvers/limx/linear_solver.py`: add opt-in relative residual stopping and final residual reporting without changing fixed-iteration behavior.
- `newton/_src/solvers/limx/solver_newton.py`: add nested line-search/diagnostic types, total objective evaluation, gradient checks, candidate positions, Armijo backtracking, and failure semantics.
- `newton/_src/solvers/limx/constraints/__init__.py`: export both new constraints internally.
- `newton/_src/solvers/limx/__init__.py`: expose both constraints through the solver package.
- `newton/tests/test_solver_limx.py`: cover anchor energy, PCG relative tolerance, objective validation, compatibility, diagnostics, and Armijo behavior.
- `docs/api/newton_solvers.rst`: generated public API entries.
- `README.md`: register the example and image.
- `CHANGELOG.md`: add one user-facing `[Unreleased] / Added` entry.

---

### Task 1: Shared tetrahedral infrastructure and quadratic baseline

**Files:**
- Create: `newton/_src/solvers/limx/constraints/tetrahedron_elastic_common.py`
- Create: `newton/_src/solvers/limx/constraints/tetrahedron_linear_elastic.py`
- Create: `newton/tests/test_constraint_tetrahedron_elastic.py`
- Modify: `newton/_src/solvers/limx/constraints/anchor.py`
- Modify: `newton/tests/test_solver_limx.py:133-179`

**Interfaces:**
- Produces: `_TetrahedronElasticConstraintBase`, `vec9`, `mat99`, `_deformation_gradient()`, `_material_gradient()`, `_map_hessian_block()`, `_project_psd()`, and `ConstraintTetrahedronLinearElastic`.
- Produces on energy-aware constraints: `accumulate_energy(positions: wp.array[wp.vec3], output: wp.array[float], invalid_count: wp.array[int]) -> None`.
- Produces on tetrahedral constraints: `minimum_determinant(positions: wp.array[wp.vec3]) -> float` for diagnostic host reads.
- Consumes: existing `BlockCsrBuilder`, `BlockCsrMatrix`, and `ConstraintAnchor` force/Hessian conventions.

- [ ] **Step 1: Write failing construction and anchor-energy tests**

Add imports for the not-yet-created class and tests with exact rest geometry:

```python
REST_POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)
TETS = [(0, 1, 2, 3)]
DM_INV = [wp.mat33(1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0)]

class TestConstraintTetrahedronLinearElastic(unittest.TestCase):
    def test_rest_state_has_zero_energy_and_force(self):
        """Keep quadratic tetrahedra force-free at the rest state."""
        constraint = ConstraintTetrahedronLinearElastic(
            TETS, DM_INV, [4.0], [7.0], 4, "cuda:0"
        )
        positions = wp.array(REST_POSITIONS, dtype=wp.vec3, device="cuda:0")
        force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
        energy = wp.zeros(1, dtype=float, device="cuda:0")
        invalid = wp.zeros(1, dtype=int, device="cuda:0")
        constraint.accumulate_force(positions, force)
        constraint.accumulate_energy(positions, energy, invalid)
        np.testing.assert_allclose(force.numpy(), 0.0, atol=1.0e-6)
        self.assertAlmostEqual(float(energy.numpy()[0]), 0.0, places=6)
        self.assertEqual(int(invalid.numpy()[0]), 0)

    def test_rejects_invalid_material_and_rest_data(self):
        """Reject malformed tetrahedra, rest matrices, and Lamé parameters."""
        with self.assertRaisesRegex(ValueError, "shear moduli"):
            ConstraintTetrahedronLinearElastic(TETS, DM_INV, [0.0], [7.0], 4, "cuda:0")
        with self.assertRaisesRegex(ValueError, "Lamé"):
            ConstraintTetrahedronLinearElastic(TETS, DM_INV, [4.0], [-1.0], 4, "cuda:0")

class TestConstraintAnchorEnergy(unittest.TestCase):
    def test_accumulates_quadratic_anchor_energy(self):
        """Accumulate one-half stiffness times squared anchor displacement."""
        anchor = ConstraintAnchor([0], [wp.vec3(1.0, 0.0, 0.0)], [10.0], 1, "cuda:0")
        positions = wp.array([[1.5, 0.0, 0.0]], dtype=wp.vec3, device="cuda:0")
        energy = wp.zeros(1, dtype=float, device="cuda:0")
        invalid = wp.zeros(1, dtype=int, device="cuda:0")
        anchor.accumulate_energy(positions, energy, invalid)
        self.assertAlmostEqual(float(energy.numpy()[0]), 1.25, places=6)
        self.assertEqual(int(invalid.numpy()[0]), 0)
```

- [ ] **Step 2: Run the focused tests and verify red state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_constraint_tetrahedron_elastic.py -k LinearElastic
uv run --extra dev -m newton.tests -p test_solver_limx.py -k test_accumulates_quadratic_anchor_energy
```

Expected: FAIL because `tetrahedron_linear_elastic` and `ConstraintAnchor.accumulate_energy()` do not exist.

- [ ] **Step 3: Implement common geometry, validation, CSR binding, and anchor energy**

Implement `_TetrahedronElasticConstraintBase.__init__()` with parameters
`tetrahedron_indices: Sequence[tuple[int, int, int, int]]`,
`inverse_rest_matrices: Sequence[wp.mat33]`,
`shear_moduli: Sequence[float]`, `lame_parameters: Sequence[float]`,
`particle_count: int`, and `device: Any`. Implement the exact public-like
methods `append_hessian_structure(builder: BlockCsrBuilder) -> None`,
`bind_hessian(matrix: BlockCsrMatrix) -> None`, and
`minimum_determinant(positions: wp.array[wp.vec3]) -> float`.

Validation must require equal nonzero element-array lengths, four distinct in-range indices, finite nonsingular rest matrices with positive `V0 = 1 / (6 det(Dm^-1))`, finite positive shear moduli, and finite nonnegative Lamé first parameters. Store host tuples plus Warp arrays and all sixteen block indices per tetrahedron.

Add the anchor kernel:

```python
@wp.kernel
def _accumulate_anchor_energy(indices, targets, stiffnesses, positions, energy):
    constraint = wp.tid()
    delta = positions[indices[constraint]] - targets[constraint]
    wp.atomic_add(energy, 0, 0.5 * stiffnesses[constraint] * wp.dot(delta, delta))
```

`ConstraintAnchor.accumulate_energy()` validates positions, one-float energy output, and one-int invalid output; anchors never increment invalid count.

- [ ] **Step 4: Implement the quadratic energy, force, and constant Hessian**

Use the approved equations exactly:

```text
epsilon = 0.5 * (F + F^T) - I
Psi = V0 * (mu * epsilon:epsilon + 0.5 * lambda * trace(epsilon)^2)
P = V0 * (2 mu epsilon + lambda trace(epsilon) I)
dP[dF] = V0 * (mu (dF + dF^T) + lambda trace(dF) I)
```

Implement `accumulate_energy()`, `accumulate_force()`, and `accumulate_force_and_hessian()` on `ConstraintTetrahedronLinearElastic`. Map `P * material_gradient` to particle gradients, subtract it from force output, and map the constant `9 x 9` material Hessian through all sixteen particle blocks.

- [ ] **Step 5: Add derivative, assembly, and PSD tests**

Use a deformed positive-volume tetrahedron and centered finite differences:

```python
DEFORMED_POSITIONS = np.asarray(
    [[0.10, -0.05, 0.02], [1.18, 0.08, -0.03],
     [0.04, 0.91, 0.12], [-0.06, 0.10, 1.09]], dtype=np.float32
)

def test_force_matches_negative_energy_gradient(self):
    """Match quadratic tetrahedral force to centered energy differences."""
    analytical_force = assemble_force(DEFORMED_POSITIONS)
    numerical_force = np.empty((4, 3), dtype=np.float32)
    step = 1.0e-3
    for vertex in range(4):
        for axis in range(3):
            positive = DEFORMED_POSITIONS.copy()
            negative = DEFORMED_POSITIONS.copy()
            positive[vertex, axis] += step
            negative[vertex, axis] -= step
            numerical_force[vertex, axis] = -(
                evaluate_energy(positive) - evaluate_energy(negative)
            ) / (2.0 * step)
    np.testing.assert_allclose(analytical_force, numerical_force, rtol=2e-3, atol=2e-3)

def test_assembled_hessian_is_constant_symmetric_psd(self):
    """Assemble a constant symmetric PSD quadratic tetrahedral Hessian."""
    rest_hessian = assemble_dense_hessian(REST_POSITIONS)
    deformed_hessian = assemble_dense_hessian(DEFORMED_POSITIONS)
    np.testing.assert_allclose(rest_hessian, deformed_hessian, atol=1e-6)
    np.testing.assert_allclose(rest_hessian, rest_hessian.T, atol=1e-6)
    self.assertGreaterEqual(float(np.linalg.eigvalsh(rest_hessian).min()), -1e-5)
    translation = np.tile([0.3, -0.2, 0.7], 4)
    np.testing.assert_allclose(rest_hessian @ translation, 0.0, atol=1e-5)
```

Implement `assemble_force()`, `evaluate_energy()`, and
`assemble_dense_hessian()` as test helpers immediately above the test class;
each helper builds Warp buffers on the requested device and converts the one
result being asserted with `.numpy()`.

- [ ] **Step 6: Run focused tests and verify green state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_constraint_tetrahedron_elastic.py -k LinearElastic
uv run --extra dev -m newton.tests -p test_solver_limx.py -k Anchor
```

Expected: PASS with all energy, derivative, assembly, and validation assertions green.

- [ ] **Step 7: Commit the quadratic baseline**

```bash
git add newton/_src/solvers/limx/constraints/anchor.py \
  newton/_src/solvers/limx/constraints/tetrahedron_elastic_common.py \
  newton/_src/solvers/limx/constraints/tetrahedron_linear_elastic.py \
  newton/tests/test_constraint_tetrahedron_elastic.py \
  newton/tests/test_solver_limx.py
git commit -m "Add LIMX linear tetrahedral elasticity"
```

---

### Task 2: Logarithmic Neo-Hookean tetrahedral constraint

**Files:**
- Create: `newton/_src/solvers/limx/constraints/tetrahedron_neo_hookean.py`
- Modify: `newton/tests/test_constraint_tetrahedron_elastic.py`

**Interfaces:**
- Consumes: `_TetrahedronElasticConstraintBase`, `mat99`, `_deformation_gradient()`, `_material_gradient()`, `_map_hessian_block()`, and `_project_psd()` from Task 1.
- Produces: `ConstraintTetrahedronNeoHookean` with the same constructor and static-constraint methods as the quadratic class.
- Produces private testable functions `_neo_hookean_energy()`, `_neo_hookean_gradient()`, and `_neo_hookean_hessian()` using column-major `vec(F)`.

- [ ] **Step 1: Write failing logarithmic energy and derivative tests**

Add a NumPy reference that rejects nonpositive determinants:

```python
def neo_hookean_reference(f, mu, lame, volume):
    determinant = np.linalg.det(f)
    if determinant <= 0.0:
        return np.inf
    log_j = np.log(determinant)
    return volume * (
        0.5 * mu * (np.sum(f * f) - 3.0)
        - mu * log_j
        + 0.5 * lame * log_j * log_j
    )
```

Add tests:

```python
class TestConstraintTetrahedronNeoHookean(unittest.TestCase):
    def test_rest_state_has_zero_energy_and_force(self):
        """Keep logarithmic Neo-Hookean tetrahedra stress-free at rest."""

    def test_force_matches_negative_energy_gradient(self):
        """Match logarithmic Neo-Hookean force to centered energy differences."""

    def test_exact_material_hessian_matches_gradient_difference(self):
        """Match the complete unprojected material Hessian to gradient differences."""

    def test_inverted_state_is_invalid_without_nan_energy(self):
        """Reject nonpositive determinants before logarithm or inverse evaluation."""
```

The Hessian test perturbs all nine entries of a positive nontrivial `F` by `1e-3`, compares columns against gradient differences, and uses `rtol=3e-3`, `atol=3e-3` for fp32 kernels.

- [ ] **Step 2: Run the focused tests and verify red state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_constraint_tetrahedron_elastic.py -k NeoHookean
```

Expected: FAIL because `ConstraintTetrahedronNeoHookean` and its material functions do not exist.

- [ ] **Step 3: Implement standard compressible Neo-Hookean material functions**

Implement only for `J > 0`:

```text
q = lambda log(J) - mu
Psi = V0 [0.5 mu (F:F - 3) - mu log(J) + 0.5 lambda log(J)^2]
P = V0 [mu F + q F^-T]
H_(i,a),(j,b) = V0 [
    mu delta_ij delta_ab
    + lambda A_(i,a) A_(j,b)
    - q A_(i,b) A_(j,a)
], A = F^-T.
```

The energy kernel must branch on `J <= 0` or non-finite `J`, atomically increment `invalid_count`, and return before calling `wp.log()` or dividing by `J`. Use the cofactor divided by `J` for `F^-T` after the positive-domain check.

- [ ] **Step 4: Assemble exact force and PSD-projected Hessian**

In `accumulate_force_and_hessian()`:

```python
hessian_exact = _neo_hookean_hessian(deformation, mu, lame, rest_volume)
hessian_psd = _project_psd(hessian_exact)
```

Keep the force from the exact gradient, map all sixteen projected blocks, and
do not project each `3 x 3` block independently. The objective-aware solver
checks validity before assembly. As a defensive fallback, an assembly kernel
that nevertheless receives `J <= 0` returns before writing that element's
force or Hessian, so it cannot contaminate global buffers with NaNs.

- [ ] **Step 5: Add PSD, balance, translation-nullspace, and invalid-domain tests**

```python
def test_projected_hessian_matches_numpy_eigh(self):
    """Match full material-space eigenvalue clamping and reconstruction."""
    exact = evaluate_exact_material_hessian(DEFORMATION)
    eigenvalues, eigenvectors = np.linalg.eigh(exact)
    expected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    np.testing.assert_allclose(evaluate_projected_hessian(DEFORMATION), expected, rtol=2e-3, atol=2e-3)

def test_assembled_hessian_is_symmetric_psd_with_translation_nullspace(self):
    """Preserve symmetry, PSD behavior, and rigid translation modes after mapping."""
    force, hessian = assemble_neo_hookean(DEFORMED_POSITIONS)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-5)
    self.assertGreaterEqual(float(np.linalg.eigvalsh(hessian).min()), -2e-4)
    translation = np.tile([0.3, -0.2, 0.7], 4)
    np.testing.assert_allclose(hessian @ translation, 0.0, atol=2e-4)

def test_minimum_determinant_reports_current_tetrahedron_state(self):
    """Report the smallest deformation determinant for diagnostics."""
    expected = np.linalg.det(
        np.column_stack(
            [DEFORMED_POSITIONS[i] - DEFORMED_POSITIONS[0] for i in (1, 2, 3)]
        )
    )
    self.assertAlmostEqual(constraint.minimum_determinant(positions), expected, places=5)
```

Define `DEFORMATION`, `evaluate_exact_material_hessian()`,
`evaluate_projected_hessian()`, and `assemble_neo_hookean()` as concrete test
helpers backed by single-purpose Warp test kernels in this test module.

- [ ] **Step 6: Run focused CUDA material tests and verify green state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_constraint_tetrahedron_elastic.py -k NeoHookean
```

Expected: PASS; no test imports or calls the VBD stable Neo-Hookean function.

- [ ] **Step 7: Commit the logarithmic material**

```bash
git add newton/_src/solvers/limx/constraints/tetrahedron_neo_hookean.py \
  newton/tests/test_constraint_tetrahedron_elastic.py
git commit -m "Add LIMX logarithmic Neo-Hookean energy"
```

---

### Task 3: Relative-residual PCG diagnostics

**Files:**
- Modify: `newton/_src/solvers/limx/linear_solver.py:90-184`
- Modify: `newton/tests/test_solver_limx.py:1008-1084`

**Interfaces:**
- Produces: optional `relative_tolerance: float | None = None` on `PcgSolver.solve()` without removing the existing absolute `tolerance` argument.
- Produces: read-only `PcgSolver.last_iterations: int` and `PcgSolver.last_relative_residual: float | None` after a solve.
- Preserves: fixed-iteration calls perform the requested count and do not add host residual checks.

- [ ] **Step 1: Write failing relative-tolerance and compatibility tests**

```python
def test_relative_tolerance_stops_and_reports_residual(self):
    """Stop PCG relative to its initial residual and report the final ratio."""
    operator = self.make_operator()
    rhs = wp.array(
        [wp.vec3(9.75, -23.0, 6.0), wp.vec3(3.0, 50.0, -16.5)],
        dtype=wp.vec3,
        device="cpu",
    )
    solution = wp.zeros(2, dtype=wp.vec3, device="cpu")
    solver = PcgSolver(2, "cpu")
    executed = solver.solve(
        operator, rhs, solution, iterations=100,
        relative_tolerance=1.0e-6, check_interval=1,
    )
    self.assertEqual(solver.last_iterations, executed)
    self.assertLess(executed, 100)
    self.assertLessEqual(solver.last_relative_residual, 1.0e-6)

def test_fixed_iteration_path_does_not_report_relative_residual(self):
    """Preserve synchronization-free fixed-count PCG behavior by default."""
    executed = solver.solve(operator, rhs, solution, iterations=7)
    self.assertEqual(executed, 7)
    self.assertEqual(solver.last_iterations, 7)
    self.assertIsNone(solver.last_relative_residual)
```

Also assert that supplying both absolute `tolerance` and `relative_tolerance` raises `ValueError`.

- [ ] **Step 2: Run PCG tests and verify red state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k PcgSolver
```

Expected: FAIL because the new keyword and diagnostic properties do not exist.

- [ ] **Step 3: Implement relative residual initialization and checks**

After constructing the actual initial residual, compute `r0_squared` once only when `relative_tolerance` is set. Stop immediately with ratio zero for a zero initial residual. At each `check_interval`, compare

```text
r_squared <= relative_tolerance^2 * r0_squared.
```

Store `last_iterations` and `last_relative_residual = sqrt(r_squared / r0_squared)` before every return. Keep `last_relative_residual = None` and avoid `.numpy()` entirely when neither tolerance mode is active.

- [ ] **Step 4: Run PCG tests and verify green state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k PcgSolver
```

Expected: PASS, including all pre-existing absolute-tolerance and warm-start tests.

- [ ] **Step 5: Commit PCG diagnostics**

```bash
git add newton/_src/solvers/limx/linear_solver.py newton/tests/test_solver_limx.py
git commit -m "Report relative LIMX PCG convergence"
```

---

### Task 4: Objective-aware SolverLIMX and Armijo backtracking

**Files:**
- Modify: `newton/_src/solvers/limx/solver_newton.py:23-254`
- Modify: `newton/tests/test_solver_limx.py:2599-2950`

**Interfaces:**
- Consumes: static `accumulate_energy()` protocol from Tasks 1-2 and relative PCG from Task 3.
- Produces nested immutable `SolverLIMX.LineSearch` and `SolverLIMX.IterationDiagnostics` types.
- Adds constructor keywords `line_search`, `linear_tolerance`, `nonlinear_tolerance`, and `record_diagnostics`, all defaulting to disabled/`None`.
- Produces `last_step_diagnostics: tuple[SolverLIMX.IterationDiagnostics, ...]`.

- [ ] **Step 1: Write failing configuration, compatibility, and diagnostics tests**

Define the exact nested records in test expectations:

```python
line_search = SolverLIMX.LineSearch(
    armijo_coefficient=1.0e-4,
    contraction_factor=0.5,
    max_backtracks=12,
)

def test_defaults_preserve_full_step_fixed_iteration_behavior(self):
    """Preserve legacy LIMX iteration and synchronization behavior by default."""
    solver = SolverLIMX(model, [], nonlinear_iterations=2, linear_iterations=3)
    self.assertIsNone(solver.line_search)
    self.assertIsNone(solver.linear_tolerance)
    self.assertIsNone(solver.nonlinear_tolerance)
    self.assertFalse(solver.record_diagnostics)

def test_rejects_objective_mode_without_constraint_energy(self):
    """Reject incomplete objectives instead of silently omitting a constraint."""
    with self.assertRaisesRegex(ValueError, "accumulate_energy"):
        SolverLIMX(model, [energy_unaware_constraint], record_diagnostics=True)

def test_rejects_line_search_with_dynamic_operator(self):
    """Reject line search for unsupported matrix-free dynamic objectives."""
    with self.assertRaisesRegex(ValueError, "dynamic"):
        SolverLIMX(model, [anchor], line_search=line_search,
                   dynamic_operator=recording_dynamic_operator)
```

- [ ] **Step 2: Write a failing quadratic one-step convergence test**

Build one tetrahedron with one anchored face, zero gravity, and a positive deformed input state. Use `ConstraintTetrahedronLinearElastic`, `linear_tolerance=1e-7`, `nonlinear_tolerance=1e-5`, `record_diagnostics=True`, and at most three Newton iterations. Assert:

```python
def test_quadratic_objective_reaches_linear_accuracy_after_one_step(self):
    """Reduce a quadratic implicit objective to the PCG accuracy floor in one Newton step."""
    solver.step(state_in, state_out, None, None, 0.03)
    records = solver.last_step_diagnostics
    self.assertGreaterEqual(len(records), 1)
    self.assertEqual(records[0].step_length, 1.0)
    if len(records) > 1:
        self.assertLessEqual(records[1].relative_gradient_norm, 2.0e-5)
```

- [ ] **Step 3: Write a failing Armijo distorted-tetrahedron test**

Use a deterministic single tetrahedron with rest vertices
`[(0,0,0), (1,0,0), (0,1,0), (0,0,1)]`, anchor vertex zero at `1.0e8 N/m`,
start current vertices at
`[(0,0,0), (0.22,0.04,0), (0.03,0.35,0.02), (0.01,0.02,0.18)]`,
and assign free-vertex velocities
`[(0,0,0), (-4,1,0), (1,-5,0), (0,1,-4)] m/s`. Use `dt=0.05 s`,
`E=1.0e6 Pa`, and `nu=0.3`. The initial `[2, 3, 4]` multiplier probe did not
make the full step unsafe. A subsequent deterministic integer sweep over the
same velocity direction found that multipliers `1` through `179` decreased the
objective and `180` was the first to increase it (`+3936 J` in the approved
fp32 fixture). Freeze `180` in the regression rather than searching at test
time. First run the no-search solver and assert from its diagnostic record
that the full candidate is either invalid or has
`objective_after > objective_before`. From the same frozen `state_in`, compare
the Armijo solver and assert:

```python
def test_armijo_backtracks_to_positive_sufficient_decrease(self):
    """Backtrack an unsafe Neo-Hookean step to a positive sufficient decrease."""
    armijo.step(state_in, state_out, None, None, 0.05)
    record = armijo.last_step_diagnostics[0]
    self.assertGreater(record.backtracks, 0)
    self.assertGreater(record.minimum_determinant, 0.0)
    self.assertLessEqual(
        record.objective_after,
        record.objective_before
        + 1.0e-4 * record.step_length * record.directional_derivative,
    )
```

The red-green evidence must include the frozen multiplier and the observed
full-step status so later runs do not perform a parameter search or weaken the
assertion.

- [ ] **Step 4: Run solver tests and verify red state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py \
  -k "test_defaults_preserve_full_step_fixed_iteration_behavior or test_quadratic_objective_reaches_linear_accuracy_after_one_step or test_armijo_backtracks_to_positive_sufficient_decrease"
```

Expected: FAIL because the nested types, objective path, and diagnostics do not exist.

- [ ] **Step 5: Implement nested configuration and diagnostic types**

Use frozen dataclasses nested in `SolverLIMX`:

```python
@dataclass(frozen=True)
class LineSearch:
    armijo_coefficient: float = 1.0e-4
    contraction_factor: float = 0.5
    max_backtracks: int = 12

@dataclass(frozen=True)
class IterationDiagnostics:
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
```

Validate finite coefficients, `0 < armijo_coefficient < 1`, `0 < contraction_factor < 1`, nonnegative backtracks, and positive tolerances. Allocate objective, invalid-count, candidate-position, and scalar-reduction buffers only when the opt-in path needs them.

- [ ] **Step 6: Implement total objective and gradient diagnostics**

Add inertia energy

```text
sum_i 0.5 * mass_i / dt^2 * ||x_i - inertia_position_i||^2
```

then call every constraint's `accumulate_energy()`. Return `(objective, invalid_count)` only after the scalar arrays copy to host. Compute gradient norm from `rhs = -gradient`. Capture the first gradient norm as the per-step normalization denominator.

The objective-aware constructor must verify every static constraint has `accumulate_energy` and that the dynamic operator is `EmptyDynamicConstraintOperator`. Diagnostic-only no-search runs use the same complete objective validation as Armijo runs.

- [ ] **Step 7: Implement full-step diagnostics and Armijo backtracking**

For no search, apply `alpha=1`, evaluate the candidate only when objective mode is active, record invalid/non-finite status, and terminate without repair on failure.

For Armijo:

```python
directional_derivative = -dot(rhs, increment)  # g dot p, because rhs = -g
alpha = 1.0
for backtracks in range(config.max_backtracks + 1):
    candidate = iterate + alpha * increment
    candidate_objective, invalid = evaluate_objective(candidate, dt)
    if invalid == 0 and candidate_objective <= (
        objective_before
        + config.armijo_coefficient * alpha * directional_derivative
    ):
        copy(candidate, iterate)
        break
    alpha *= config.contraction_factor
else:
    status = "line_search_failed"
```

Stop with `non_descent_direction` when `directional_derivative >= 0`. Stop before the linear solve when relative gradient norm reaches `nonlinear_tolerance`. Preserve the old warm-start rule and call signature exactly when `linear_tolerance is None`, so existing recording test doubles need no new keyword.

- [ ] **Step 8: Run solver and regression tests and verify green state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k "PcgSolver or SolverLIMX or Anchor"
uv run --extra dev -m newton.tests -p test_constraint_tetrahedron_elastic.py
```

Expected: PASS. In particular, the existing previous-frame PCG warm-start test and dynamic-contact tests remain unchanged.

- [ ] **Step 9: Commit objective-aware Newton**

```bash
git add newton/_src/solvers/limx/solver_newton.py newton/tests/test_solver_limx.py
git commit -m "Add Armijo search to LIMX Newton steps"
```

---

### Task 5: Public exports and generated API

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/__init__.py:6-32`
- Modify: `newton/_src/solvers/limx/__init__.py:6-47`
- Modify: `newton/tests/test_constraint_tetrahedron_elastic.py`
- Modify: `newton/tests/test_solver_limx.py:2926-2933`
- Modify: `docs/api/newton_solvers.rst`

**Interfaces:**
- Consumes: both material classes from Tasks 1-2 and nested solver types from Task 4.
- Produces: `newton.solvers.ConstraintTetrahedronLinearElastic` and `newton.solvers.ConstraintTetrahedronNeoHookean`.

- [ ] **Step 1: Write failing public-export tests**

```python
def test_public_exports(self):
    """Expose LIMX tetrahedral materials through the public solver module."""
    self.assertIs(
        newton.solvers.ConstraintTetrahedronLinearElastic,
        ConstraintTetrahedronLinearElastic,
    )
    self.assertIs(
        newton.solvers.ConstraintTetrahedronNeoHookean,
        ConstraintTetrahedronNeoHookean,
    )
```

- [ ] **Step 2: Run the export test and verify red state**

Run:

```bash
uv run --extra dev -m newton.tests -p test_constraint_tetrahedron_elastic.py -k public_exports
```

Expected: FAIL because the symbols are not yet in LIMX `__all__`.

- [ ] **Step 3: Add ordered internal and public exports**

Import and add both names alphabetically in the constraint and LIMX package `__all__` lists. Do not edit `newton/solvers.py`; its lazy public export derives from `_src.solvers.__all__`.

- [ ] **Step 4: Generate and verify API documentation**

Run:

```bash
uv run docs/generate_api.py
rg -n "ConstraintTetrahedron(LinearElastic|NeoHookean)|SolverLIMX" docs/api/newton_solvers.rst
```

Expected: both constraints appear in `docs/api/newton_solvers.rst`, and generated docs contain no `newton._src` references.

- [ ] **Step 5: Run export and API tests**

Run:

```bash
uv run --extra dev -m newton.tests -p test_constraint_tetrahedron_elastic.py -k public_exports
uv run --extra dev -m newton.tests -p test_generate_api.py
```

Expected: PASS.

- [ ] **Step 6: Commit public API exposure**

```bash
git add newton/_src/solvers/limx/constraints/__init__.py \
  newton/_src/solvers/limx/__init__.py \
  newton/tests/test_constraint_tetrahedron_elastic.py \
  newton/tests/test_solver_limx.py docs/api/newton_solvers.rst
git commit -m "Expose LIMX tetrahedral materials"
```

---

### Task 6: Visual Neo-Hookean cantilever example

**Files:**
- Create: `newton/examples/softbody/example_softbody_limx_neo_hookean_beam.py`
- Create: `newton/tests/test_example_softbody_limx_neo_hookean_beam.py`

**Interfaces:**
- Consumes only public `newton.solvers` symbols from Tasks 4-5.
- Produces: `Example`, `create_cantilever_model(device)`, `create_material_constraints(model, rest_positions, material)`, and standard example parser behavior.
- Produces CLI `--line-search/--no-line-search`, defaulting to line search enabled for the logarithmic material.

- [ ] **Step 1: Write a failing CUDA example-construction test**

```python
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
```

Define the local helper exactly as:

```python
def make_args(*, line_search: bool):
    return argparse.Namespace(
        line_search=line_search,
        convergence_study=False,
        output_directory=".",
    )
```

The example constructor reads no viewer/parser fields outside this namespace;
standard launch-only arguments remain handled by `newton.examples.init()`.

- [ ] **Step 2: Run the construction test and verify red state**

Run:

```bash
uv run --extra dev -m newton.tests \
  -p test_example_softbody_limx_neo_hookean_beam.py \
  -k test_builds_approved_cantilever_configuration
```

Expected: FAIL because the example module does not exist.

- [ ] **Step 3: Implement the cantilever and public material setup**

Use `ModelBuilder.add_soft_grid()` with:

```python
builder.add_soft_grid(
    pos=wp.vec3(0.0, -0.05, 0.75),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0),
    dim_x=12, dim_y=2, dim_z=2,
    cell_x=0.05, cell_y=0.05, cell_z=0.05,
    density=1000.0,
    k_mu=0.0, k_lambda=0.0, k_damp=0.0,
    fix_left=False,
)
```

Convert `E=1.0e6`, `nu=0.3` with

```text
mu = E / (2 (1 + nu))
lambda = E nu / ((1 + nu) (1 - 2 nu)).
```

Anchor the leftmost nine particles at `1.0e8 N/m`; create `ConstraintTetrahedronNeoHookean` from public APIs; use at most 20 Newton iterations, 256 PCG iterations, relative tolerances from the spec, no dynamic operator, and `velocity_damping=1.0`.

- [ ] **Step 4: Implement visual lifecycle and physical assertions**

`step()` clears forces, applies viewer forces, calls the solver once at `0.01 s`, swaps states, and advances time. `test_post_step()` checks finite positions/velocities and positive current tetrahedron volumes. `test_final()` checks anchors within `2.0e-3 m` and the free-end mean height at least `2.0e-3 m` below its initial value.

- [ ] **Step 5: Add one-frame and rollout tests**

```python
def test_one_step_keeps_state_finite_and_positive(self):
    """Advance one CUDA step without inversion or non-finite state."""

def test_free_end_falls_while_left_face_stays_anchored(self):
    """Drop the free cantilever end while retaining the anchored face."""
    for _ in range(20):
        example.step()
        example.test_post_step()
    example.test_final()

def test_no_line_search_selects_full_steps(self):
    """Disable Armijo explicitly for the visual comparison variant."""
```

- [ ] **Step 6: Run focused CUDA example tests and a null-viewer smoke run**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_softbody_limx_neo_hookean_beam.py
uv run --extra examples -m newton.examples softbody_limx_neo_hookean_beam \
  --device cuda:0 --viewer null --num-frames 20 --line-search
```

Expected: PASS; the run reports no invalid determinant or line-search failure.

- [ ] **Step 7: Commit the visual example**

```bash
git add newton/examples/softbody/example_softbody_limx_neo_hookean_beam.py \
  newton/tests/test_example_softbody_limx_neo_hookean_beam.py
git commit -m "Add LIMX Neo-Hookean beam example"
```

---

### Task 7: Deterministic convergence CSV and PNG study

**Files:**
- Modify: `newton/examples/softbody/example_softbody_limx_neo_hookean_beam.py`
- Modify: `newton/tests/test_example_softbody_limx_neo_hookean_beam.py`

**Interfaces:**
- Produces: `run_convergence_study(device: str, output_directory: Path, *, checkpoint_time: float = 0.30, time_steps: tuple[float, ...] = (0.01, 0.03, 0.05), max_newton_iterations: int = 20) -> list[dict[str, float | int | str]]`.
- Produces CLI `--convergence-study` and `--output-directory` without repurposing Newton's existing `--benchmark` option.
- Writes `limx_neo_hookean_convergence.csv` and `limx_neo_hookean_convergence.png`.

- [ ] **Step 1: Write failing reduced-study output tests**

Use `tempfile.TemporaryDirectory()` and reduced checkpoint/iteration parameters exposed only as keyword-only test hooks:

```python
def test_convergence_study_writes_complete_csv(self):
    """Write one diagnostic row per recorded Newton iteration."""
    rows = run_convergence_study(
        "cuda:0", Path(temp_dir), checkpoint_time=0.02,
        time_steps=(0.01,), max_newton_iterations=3,
    )
    csv_path = Path(temp_dir) / "limx_neo_hookean_convergence.csv"
    self.assertTrue(csv_path.is_file())
    self.assertEqual({row["method"] for row in rows},
                     {"quadratic", "neo_hookean_full", "neo_hookean_armijo"})
    required = {
        "time_step", "method", "iteration", "objective_before",
        "objective_after", "relative_gradient_norm", "step_length",
        "backtracks", "linear_iterations", "linear_relative_residual",
        "minimum_determinant", "status",
    }
    self.assertTrue(required.issubset(rows[0]))

def test_convergence_study_writes_nonempty_png(self):
    """Render the six-panel convergence comparison to a nonempty PNG."""
    self.assertGreater(png_path.stat().st_size, 1000)
```

- [ ] **Step 2: Run output tests and verify red state**

Run:

```bash
uv run --extra dev --extra examples -m newton.tests \
  -p test_example_softbody_limx_neo_hookean_beam.py \
  -k convergence_study
```

Expected: FAIL because the study functions and files do not exist.

- [ ] **Step 3: Implement deterministic checkpoint generation**

Build one line-search Neo-Hookean example, run exactly `round(checkpoint_time / 0.01)` reference steps, and copy both particle positions and velocities to NumPy. Rebuild each comparison solver/model, assign the identical checkpoint arrays, and use the same anchors/rest mesh/material constants. Do not reuse mutable solver state between methods.

- [ ] **Step 4: Implement the nine comparison solves and reference objectives**

For each requested time step run:

```text
quadratic
neo_hookean_full
neo_hookean_armijo
```

Use a tighter-tolerance Armijo solve of the same material and time step as one
candidate for the normalized-gap baseline. Because fp32 anchor cancellation
can stop this solve before its requested gradient tolerance, define the
baseline as the lowest finite objective observed across the tight solve and
the measured runs. Record the baseline source and the tight solve's terminal
status and relative gradient norm. Preserve raw values and terminal failure
rows. Compute normalized gaps within each material/time-step objective only;
never subtract the quadratic baseline from Neo-Hookean energy.

- [ ] **Step 5: Implement stable CSV serialization**

Use a fixed field order beginning with `time_step`, `method`, and `iteration`,
followed by every `IterationDiagnostics` field plus `objective_baseline`,
`baseline_source`, the tight-reference objective and terminal metadata, and
`objective_gap`.
Convert `None` linear residuals to an empty CSV field rather than the string
`None`.

- [ ] **Step 6: Implement the `2 x 3` Matplotlib plot**

Import `matplotlib.pyplot` inside the plotting function. Columns are `dt=0.01, 0.03, 0.05`; top row is relative gradient norm and bottom row is normalized objective gap, both log scale. Use fixed method colors/styles, grid, per-column titles, shared row labels, one figure legend, and annotated terminal failure markers. Save at readable DPI and close the figure.

- [ ] **Step 7: Run reduced output tests and verify green state**

Run:

```bash
uv run --extra dev --extra examples -m newton.tests \
  -p test_example_softbody_limx_neo_hookean_beam.py \
  -k convergence_study
```

Expected: PASS with a parseable CSV and nonempty PNG.

- [ ] **Step 8: Run the full approved CUDA study and inspect outputs**

Run:

```bash
study_dir=$(mktemp -d)
uv run --extra examples -m newton.examples softbody_limx_neo_hookean_beam \
  --device cuda:0 --viewer null --convergence-study \
  --output-directory "$study_dir"
python - "$study_dir/limx_neo_hookean_convergence.csv" <<'PY'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
assert rows
assert {float(row["time_step"]) for row in rows} == {0.01, 0.03, 0.05}
assert {row["method"] for row in rows} == {
    "quadratic", "neo_hookean_full", "neo_hookean_armijo"
}
for row in rows:
    assert math.isfinite(float(row["objective_before"]))
print(f"validated {len(rows)} convergence rows")
PY
```

Inspect the generated PNG visually and record, without presupposing it, whether full steps converge, oscillate, invert, or hit the cap at each `dt`.

- [ ] **Step 9: Commit the study tooling**

```bash
git add newton/examples/softbody/example_softbody_limx_neo_hookean_beam.py \
  newton/tests/test_example_softbody_limx_neo_hookean_beam.py
git commit -m "Plot LIMX Neo-Hookean convergence"
```

---

### Task 8: Documentation, changelog, screenshot, and final verification

**Files:**
- Modify: `README.md:869-899`
- Modify: `CHANGELOG.md:5-64`
- Create: `docs/images/examples/example_softbody_limx_neo_hookean_beam.jpg`
- Verify generated: `docs/api/newton_solvers.rst`

**Interfaces:**
- Consumes: finished public API, example, tests, and convergence output.
- Produces: discoverable example documentation and user-facing release note.

- [ ] **Step 1: Capture the approved `320 x 320` example image**

Run the line-search visual example on CUDA, position the camera as encoded by the example, capture the deformed cantilever after visible free-end descent, and write exactly:

```text
docs/images/examples/example_softbody_limx_neo_hookean_beam.jpg
```

Verify dimensions with an existing image tool or Pillow from the examples environment:

```bash
uv run --extra examples python - <<'PY'
from PIL import Image
path = "docs/images/examples/example_softbody_limx_neo_hookean_beam.jpg"
with Image.open(path) as image:
    assert image.size == (320, 320), image.size
PY
```

- [ ] **Step 2: Register the example in the README**

Add a Softbody Examples cell containing:

```html
<a href="https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_limx_neo_hookean_beam.py">
  <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_softbody_limx_neo_hookean_beam.jpg" alt="LIMX Neo-Hookean Beam">
</a>
```

and the exact command:

```html
<code>python -m newton.examples softbody_limx_neo_hookean_beam</code>
```

Keep every table row structurally valid with three cells.

- [ ] **Step 3: Add the user-facing changelog entry**

Insert at a random position inside `[Unreleased] / Added`, not always at the top:

```markdown
- Add logarithmic Neo-Hookean tetrahedra, optional Armijo line search and convergence diagnostics to `SolverLIMX`, and a fixed-cantilever convergence study.
```

- [ ] **Step 4: Run focused CUDA and example verification**

Run:

```bash
uv run --extra dev -m newton.tests -p test_constraint_tetrahedron_elastic.py
uv run --extra dev -m newton.tests -p test_solver_limx.py -k "PcgSolver or SolverLIMX or Anchor"
uv run --extra dev --extra examples -m newton.tests \
  -p test_example_softbody_limx_neo_hookean_beam.py
uv run --extra dev -m newton.tests -p test_generate_api.py
uv run --extra examples -m newton.examples softbody_limx_neo_hookean_beam \
  --device cuda:0 --viewer null --num-frames 100 --line-search
```

Expected: every command exits zero; the 100-frame example retains finite positive-volume tetrahedra and anchored vertices.

- [ ] **Step 5: Run repository pre-commit checks**

Run:

```bash
uvx pre-commit run -a
```

If the hooks rewrite unrelated pre-existing files, do not stage them. Use `git diff --name-only` to distinguish feature files from the preserved dirty files, then rerun checks on the feature file list after any hook fixes.

- [ ] **Step 6: Inspect scope and generated artifacts**

Run:

```bash
git status --short
git diff --check
git diff --stat
rg -n "stable Neo-Hookean|TO[D]O|TB[D]|newton\._src" \
  newton/examples/softbody/example_softbody_limx_neo_hookean_beam.py \
  docs/api/newton_solvers.rst README.md CHANGELOG.md
```

Expected: no whitespace errors, no placeholders, no private imports in the example/docs, and no accidental staging of the preserved user files.

- [ ] **Step 7: Commit documentation and final feature metadata**

```bash
git add README.md CHANGELOG.md docs/api/newton_solvers.rst \
  docs/images/examples/example_softbody_limx_neo_hookean_beam.jpg
git commit -m "Document LIMX Neo-Hookean convergence"
```

- [ ] **Step 8: Request final code review before integration**

Invoke the required `code-review` skill and review the full range from `3fb9af42` through `HEAD` on both axes:

```text
Standards: Newton API, Warp typing, tests, docs, compatibility, and scope.
Spec: every requirement in 2026-08-19-limx-neo-hookean-convergence-design.md.
```

Resolve all actionable findings, rerun the affected focused commands, and only then use `superpowers:verification-before-completion` before claiming the implementation complete.
