# LIMX Kinematic Mesh Contact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete VF/EE force and PSD Hessian contact between LIMX cloth and a kinematic Franka triangle surface, then visualize a cloth settling on the stationary wrist and gripper.

**Architecture:** A new public `ConstraintKinematicMeshContact` owns a flattened kinematic surface, cross-mesh BVHs, three frozen contact buffers, and all normal/damping/friction force and Hessian operations. Rigid vertices are transformed from body-local coordinates but never enter the LIMX unknown vector; each contact stores only cloth ids and differentiates only cloth coordinates. A separate Franka drop example composes the new constraint with existing LIMX elasticity, bending, and self-collision.

**Tech Stack:** Python, NumPy, NVIDIA Warp kernels/BVH, Newton `Model` geometry, LIMX matrix-free operators, `unittest`.

## Global Constraints

- Use discrete cloth-V/rigid-F, rigid-V/cloth-F, and cloth-E/rigid-E detection; do not substitute SDF contact.
- Every normal, damping, friction, and mollified EE term must implement force, HVP, and block diagonal consistently.
- Keep rigid geometry kinematic and outside the LIMX unknown vector in this milestone.
- Use 3 mm activation distance, 20 kN/m stiffness, 0.5 N·s/m damping, friction 0.4, and friction regularization 0.01 m/s in the example.
- Use two nonlinear iterations, 50 PCG iterations, ten substeps per 60 Hz frame, and 65,536 contacts per stencil family.
- Add no required or optional dependency; use Warp, NumPy, and stdlib only.
- Use `unittest`, give every test method a triple-double-quoted imperative docstring, and do not synchronize immediately before `.numpy()`.
- Examples and tests import only public `newton` modules, never `newton._src`.
- Preserve the existing Franka grasp scaffold; add a separate drop example.
- Run focused validation and the rendered example before broader checks.

---

### Task 1: Extract and update a kinematic triangle surface

**Files:**
- Create: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`
- Test: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`

**Interfaces:**
- Consumes: finalized `newton.Model`, selected rigid `shape_indices`, `state.body_q`, and `state.body_qd`.
- Produces:

```python
ConstraintKinematicMeshContact(
    model: Model,
    shape_indices: Sequence[int],
    thickness: float,
    stiffness: float,
    normal_damping: float,
    friction: float,
    friction_epsilon: float,
    max_contacts: int = 32768,
)
update_colliders(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector] | None = None,
) -> None
```

The class exposes device arrays `collider_positions`, `collider_velocities`,
`collider_triangles`, and `collider_edges` for diagnostics.

- [ ] **Step 1: Write failing mesh-and-box extraction tests**

Build a tiny model with one body, one triangle mesh shape, one box shape, and a separate particle triangle. Construct the constraint and assert the collider triangle count equals one plus twelve, all indices are in range, and each collider vertex has the expected body id. Add a transformed-body check:

```python
def test_update_colliders_transforms_mesh_and_box_vertices(self):
    """Transform selected mesh and box vertices into world space."""
    constraint, state = self._make_surface_fixture()
    constraint.update_colliders(state.body_q, state.body_qd)
    positions = constraint.collider_positions.numpy()
    self.assertTrue(np.isfinite(positions).all())
    self.assertTrue(np.any(positions[:, 0] > 1.0))
```

- [ ] **Step 2: Run the focused test and confirm the missing symbol failure**

Run:

```bash
uv run --extra dev -m newton.tests -k test_solver_limx_kinematic_mesh_contact
```

Expected: fail because `ConstraintKinematicMeshContact` is not exported or implemented.

- [ ] **Step 3: Implement host surface extraction**

In `kinematic_mesh_contact.py`, validate the selected shape ids. For `GeoType.MESH` and `GeoType.CONVEX_MESH`, copy `shape_source.vertices` and reshape `shape_source.indices` to triangles; apply `shape_scale` and the shape-to-body transform. For `GeoType.BOX`, emit the eight `(+/-hx, +/-hy, +/-hz)` corners and twelve consistently wound triangles. Concatenate shapes without welding vertices, record each vertex body id, and derive edges through `MeshAdjacency`.

- [ ] **Step 4: Implement the collider update kernel**

Add a Warp kernel that transforms each body-local collider point by `body_q[body]`. For nonnegative body ids compute point velocity with Newton's `velocity_at_point`; for world-attached shapes retain the pretransformed point and zero velocity. `update_colliders()` validates the body arrays and launches the kernel.

- [ ] **Step 5: Export a constructor skeleton and rerun the extraction tests**

The constructor must expose read-only diagnostic arrays and reject empty shape selection, invalid ids, unsupported geometry, malformed mesh topology, and device mismatches. Run the focused command and expect the extraction tests to pass while later contact tests remain absent.

- [ ] **Step 6: Commit the independently working surface component**

```bash
git add newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py newton/tests/test_solver_limx_kinematic_mesh_contact.py
git commit -m "Add LIMX kinematic collision surface"
```

### Task 2: Add dynamic-only frozen contact buffers

**Files:**
- Create: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py`
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`
- Modify: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`

**Interfaces:**
- Consumes: dynamic cloth ids/weights, frozen direction/depth, rigid relative velocity/displacement, stiffness, damping, and friction parameters.
- Produces: `_KinematicContactBuffer` methods `accumulate_force`, `hessian_multiply`, and `accumulate_diagonal` for arities one, two, and three.

- [ ] **Step 1: Write failing dense-Hessian consistency tests**

Inject one synthetic contact for each arity. Assemble
`H_ij = k q_i q_j outer(n, n)` independently in NumPy, then compare buffer HVP and all diagonal blocks. Include a nonnegative quadratic-form check and ensure no index outside the cloth vector is written.

```python
def test_frozen_contact_hvp_matches_dense_matrix(self):
    """Match dynamic-only contact HVP with its dense frozen Hessian."""
    dense = self._assemble_dense_rank_one(ids, weights, direction, stiffness)
    expected = (dense @ vector.reshape(-1)).reshape((-1, 3))
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)
```

- [ ] **Step 2: Verify the new tests fail**

Run the focused module and expect failure because `_KinematicContactBuffer` operations are not present.

- [ ] **Step 3: Implement normal force, HVP, and diagonal kernels**

For every active contact compute:

```text
force_i += k * depth * q_i * n
projected = sum_j q_j * dot(n, vector_j)
hvp_i += k * q_i * projected * n
diagonal_i += k * q_i^2 * outer(n, n)
```

Use a fixed maximum arity of three with an explicit stored arity per buffer. Bound all launches by `min(count, capacity)` and retain attempted-count plus overflow counters.

- [ ] **Step 4: Implement damping and regularized friction in all three paths**

Store the frozen rigid contribution to relative velocity per contact. Apply damping only when the relative normal velocity is approaching. Add its `normal_damping / dt` rank-one Hessian. Use the tangent projector and the same smooth friction regularization as LIMX self-collision; compute force, tangent HVP, and exact diagonal blocks from the same cached scalar. Cover a nonzero rigid velocity in the test so relative rather than absolute cloth motion is verified.

Interpret public `friction_epsilon` as a relative-velocity threshold [m/s] and
pass `friction_epsilon * dt` to the displacement regularizer so its units match
the step-start anchor displacement.

- [ ] **Step 5: Run focused tests and commit**

Run the focused module until normal, damping, and friction force/HVP/diagonal tests pass, then commit:

```bash
git add newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py newton/tests/test_solver_limx_kinematic_mesh_contact.py
git commit -m "Add LIMX kinematic contact Hessians"
```

### Task 3: Detect both VF directions and wire the LIMX lifecycle

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py`
- Modify: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`

**Interfaces:**
- Consumes: cloth positions/velocities/triangles, collider positions/velocities/triangles, thickness, and fixed contact parameters.
- Produces: `cloth_vertex_face_contacts` with arity one and `rigid_vertex_face_contacts` with arity three, refreshed by `prepare()`.

- [ ] **Step 1: Write failing directional VF detection tests**

Create two separated triangle fixtures. In one, place a cloth vertex above the interior of a rigid triangle and assert dynamic id `[v]`, weight `[1]`, upward normal, and `thickness-distance` depth. In the other, place a rigid vertex above a cloth triangle and assert the three cloth ids, negative barycentric weights summing to `-1`, and a force that pushes the cloth away from the rigid vertex. Add outside-projection and degenerate-triangle rejection checks.

- [ ] **Step 2: Verify VF tests fail**

Run the focused test module and expect zero contacts.

- [ ] **Step 3: Build cloth/collider triangle BVHs and detection kernels**

Allocate lower/upper bound arrays and fixed-topology `wp.Bvh` objects. During `prepare()`, update cloth and collider triangle bounds, refit both BVHs, clear contact counters, then launch cloth-V/rigid-F and rigid-V/cloth-F queries. Use absolute signed plane distance, strict interior barycentrics, two-sided normals, and the 3 mm activation distance.

- [ ] **Step 4: Wire the dynamic-constraint protocol**

Implement `begin_step()` to cache cloth anchor positions, velocities, `dt`, and collider motion. Implement `prepare()`, `accumulate_force()`, `hessian_multiply()`, and `accumulate_diagonal()` so both VF buffers contribute identical frozen terms to RHS, PCG, and the preconditioner.

- [ ] **Step 5: Run focused VF and lifecycle tests**

Add a `ConstraintGroupDynamic` fixture with the new contact and `ConstraintSelfCollision`, then verify both receive lifecycle calls and produce finite output. Run the focused module until it passes.

- [ ] **Step 6: Commit the VF milestone**

```bash
git add newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py newton/tests/test_solver_limx_kinematic_mesh_contact.py
git commit -m "Add LIMX kinematic VF contact"
```

### Task 4: Add dynamic-static EE contact and IPC mollification

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py`
- Modify: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`

**Interfaces:**
- Consumes: cloth and collider edge arrays, current/rest edge vectors, closest parameters, and rigid edge velocity.
- Produces: `edge_edge_contacts` with two dynamic cloth ids plus frozen rigid-edge data and IPC mollifier state.

- [ ] **Step 1: Write failing EE detection and derivative tests**

Use skew interior/interior edges with asymmetric closest parameters. Assert the cloth weights are `(1-s, s)`, direction points from the rigid edge to the cloth edge, and depth is correct. Add endpoint rejection. For nearly parallel edges, compare the analytic mollified HVP with a centered finite difference of the analytic force and assert the diagonal blocks are finite and positive semidefinite.

- [ ] **Step 2: Verify the EE tests fail**

Run the focused module and expect no EE candidates.

- [ ] **Step 3: Implement edge BVHs and EE narrow phase**

Derive unique edges independently for cloth and collider topology. Update/refit both edge BVHs in `prepare()`. Query collider edges from each cloth edge, compute `wp.closest_point_edge_edge`, require interior parameters, and store cloth ids/weights, rigid closest-point velocity, direction, depth, and the rest-edge mollifier threshold.

- [ ] **Step 4: Implement the dynamic-static IPC mollifier**

Reuse the current LIMX self-collision mollified residual math, treating the collider edge and its variation as constant. Differentiate the residual only through the two cloth endpoints and frozen depth direction. Implement exact mollified force, Gauss-Newton HVP, and diagonal blocks; use the same activation state to scale friction normal load.

- [ ] **Step 5: Run focused tests and commit**

Run the focused module, including the finite-difference check, then commit:

```bash
git add newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py newton/tests/test_solver_limx_kinematic_mesh_contact.py
git commit -m "Add LIMX kinematic EE contact"
```

### Task 5: Export and document the public constraint

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `CHANGELOG.md`
- Modify generated API files as produced by `docs/generate_api.py`
- Modify: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`

**Interfaces:**
- Produces: `newton.solvers.ConstraintKinematicMeshContact` with public Google-style documentation and SI units.

- [ ] **Step 1: Add a failing public-import and validation test**

Import only `newton` and instantiate `newton.solvers.ConstraintKinematicMeshContact`. Verify invalid thickness, stiffness, damping, friction, capacity, selected shapes, and unsupported geometry each raise the documented exception.

- [ ] **Step 2: Add lazy public exports and docstrings**

Update all LIMX/solver export tables. Document array shapes, units, selected geometry support, frozen-contact Hessian behavior, and the explicit `update_colliders()` lifecycle without referencing `newton._src`.

- [ ] **Step 3: Add the user-facing changelog entry and regenerate API docs**

Insert an `Added` entry at a random position in `[Unreleased]` describing kinematic VF/EE cloth contact. Run:

```bash
uv run docs/generate_api.py
```

- [ ] **Step 4: Run focused tests and commit**

```bash
uv run --extra dev -m newton.tests -k test_solver_limx_kinematic_mesh_contact
git add newton/_src/solvers CHANGELOG.md docs newton/tests/test_solver_limx_kinematic_mesh_contact.py
git commit -m "Expose LIMX kinematic mesh contact"
```

### Task 6: Build the stationary Franka drop scene

**Files:**
- Create: `newton/examples/cloth/example_cloth_limx_franka_drop.py`
- Create: `newton/tests/test_example_cloth_limx_franka_drop.py`
- Modify: `newton/tests/test_examples.py`
- Modify: `README.md`
- Create: `docs/images/examples/example_cloth_limx_franka_drop.jpg`

**Interfaces:**
- Consumes: public `ConstraintKinematicMeshContact`, existing Franka asset, square-cloth builder pattern, LIMX elasticity/bending/self-collision.
- Produces: `python -m newton.examples cloth_limx_franka_drop` and visual/contact diagnostics.

- [ ] **Step 1: Write a failing headless example test**

Register the example and run enough frames for impact. Assert finite cloth state, zero overflow, nonzero rigid-cloth contact count, and diagnostic fields `maximum_penetration`, `final_rms_speed`, and `supported`.

- [ ] **Step 2: Construct the static scene**

Load fixed-base FR3, set a palm-up joint pose, open both fingers, evaluate FK once, and select wrist/hand/finger collision shapes. Add a 0.4 m 21-by-21 active cloth 0.1 m above the target. Do not add the table or ground contact.

- [ ] **Step 3: Compose and run LIMX constraints**

Use triangle elasticity, dihedral bending, 3 mm fixed-radius self-collision, and the new kinematic contact with the exact global parameters. Use two nonlinear iterations, 50 PCG iterations, ten substeps, and update collider geometry before the LIMX step.

- [ ] **Step 4: Add diagnostics and final checks**

Track retained/attempted contacts per stencil family, overflow, maximum penetration, cloth center distance from the hand/wrist target, and a rolling one-second RMS speed. `test_final()` verifies finite values and overflow unconditionally. If settling fails, report the measured values without deleting or hiding the visual run.

After the first visual result has been shown, completion requires
`test_final()` to assert `supported` and final-second RMS speed below 0.02 m/s.
If either assertion initially fails, launch the visual scene before tuning and
report the exact failing values.

- [ ] **Step 5: Run the headless rollout**

```bash
uv run --extra dev -m newton.tests -k test_example_cloth_limx_franka_drop
uv run -m newton.examples cloth_limx_franka_drop --viewer null --num-frames 360
```

Expected: no nonfinite state or overflow; contact becomes nonzero. Record settling metrics even if the 0.02 m/s target is not yet met.

- [ ] **Step 6: Run the GL visualization and capture the screenshot**

```bash
uv run -m newton.examples cloth_limx_franka_drop --num-frames 360
```

Show the first result before tuning if it does not settle. Save a representative 320-by-320 JPG only after verifying its dimensions.

- [ ] **Step 7: Register documentation and commit**

Add the README command and screenshot, then commit:

```bash
git add newton/examples/cloth/example_cloth_limx_franka_drop.py newton/tests/test_example_cloth_limx_franka_drop.py newton/tests/test_examples.py README.md docs/images/examples/example_cloth_limx_franka_drop.jpg
git commit -m "Add LIMX Franka cloth drop scene"
```

### Task 7: Focused integration verification

**Files:**
- Modify only files changed by formatters or fixes found by the commands below.

**Interfaces:**
- Verifies: public API, contact derivatives, scene stability diagnostics, CUDA graph compatibility, documentation, and formatting.

- [ ] **Step 1: Run both focused test modules**

```bash
uv run --extra dev -m newton.tests -k test_solver_limx_kinematic_mesh_contact
uv run --extra dev -m newton.tests -k test_example_cloth_limx_franka_drop
```

Expected: all focused tests pass.

- [ ] **Step 2: Run null and GL examples**

```bash
uv run -m newton.examples cloth_limx_franka_drop --viewer null --num-frames 360
uv run -m newton.examples cloth_limx_franka_drop --num-frames 360
```

Report retained VF/EE counts, overflow, maximum penetration, support state, and final-second RMS speed. If the stability target fails, present the visual result before changing parameters.

- [ ] **Step 3: Run project pre-commit checks**

```bash
uvx pre-commit run -a
```

Expected: every hook passes. Inspect all formatter changes and preserve unrelated user edits.

- [ ] **Step 4: Review the final diff and commit verification fixes**

```bash
git diff --check
git status --short
git diff --stat
```

Commit only actual verification fixes with an imperative subject. Do not run the full Newton suite for this visual milestone.
