# LIMX Franka Box-Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the kinematically driven Franka lift active LIMX cloth using only its box collision proxies, predictive one-sided VF/EE contact, and friction at one 0.01 s step per frame.

**Architecture:** Extend `ConstraintKinematicMeshContact` so each step caches cloth and collider start states, maintains conservative swept VF/EE candidates through both Newton iterations, and evaluates signed gaps using the step-start side. Reuse the current Franka IK sequence, evaluate FK kinematically before each LIMX step, and let the existing dynamic constraint group advance only cloth.

**Tech Stack:** Python, Warp CUDA kernels, Newton LIMX, `unittest`, OpenGL example viewer.

## Global Constraints

- Use `unittest`, and give every test method a triple-double-quoted imperative docstring.
- Keep `dt = 0.01 s`, `FPS = 100`, and `SIM_SUBSTEPS = 1`.
- Use the Franka and table collision boxes; do not use visual meshes for collision.
- Do not add attachment constraints, rigid reaction forces, two-way coupling, or new dependencies.
- Keep force, HVP, and diagonal Hessian paths consistent for every active contact.
- Examples may import only Newton public modules, never `newton._src`.
- Preserve fixed-capacity CUDA-graph-compatible buffers and explicit overflow counters.

---

## File Structure

- `newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py`: swept-bound generation, signed VF/EE candidate detection, and active-depth guards for force/Hessian kernels.
- `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`: step-start/predicted arrays, swept BVH refits, and the unchanged public constraint interface.
- `newton/tests/test_solver_limx_kinematic_mesh_contact.py`: focused high-speed VF/EE and operator-consistency regression tests.
- `newton/examples/cloth/example_cloth_limx_franka.py`: active cloth, kinematic FK, box/table contact, grasp metrics, and 100 Hz rollout.
- `newton/tests/test_example_cloth_limx_franka.py`: active-cloth and physical-lift acceptance tests.
- `CHANGELOG.md`: user-visible changed behavior for the existing example and fixed kinematic contact tunneling.
- `docs/images/examples/example_cloth_limx_franka.jpg`: refreshed 320-by-320 result from the physical grasp scene.

---

### Task 1: Preserve the VF step side and stop fast face crossing

**Files:**
- Modify: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py`
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`

**Interfaces:**
- Consumes: existing `ConstraintKinematicMeshContact.begin_step(positions, velocities, dt)` and `prepare(positions)`.
- Produces: internal step-start and predicted cloth/collider arrays; swept triangle bounds; one-sided cloth-V/rigid-F and rigid-V/cloth-F depths without changing the public constructor.

- [ ] **Step 1: Write the failing VF crossing tests**

Add a fixture parameter for cloth velocity and a test that begins 10 mm above a rigid triangle, predicts a 20 mm downward move during `dt=0.01`, and calls `prepare()` at a crossed iterate 10 mm below the face:

```python
def test_keeps_cloth_vertex_contact_on_step_start_side_after_crossing(self):
    """Keep a swept cloth vertex on its step-start side of a rigid face."""
    constraint, state = self._make_vf_fixture(
        cloth_positions=((0.0, 0.0, 0.01), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0)),
        rigid_vertices=((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        cloth_velocities=((0.0, 0.0, -2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        dt=0.01,
        prepare=False,
    )
    crossed = wp.array(
        ((0.0, 0.0, -0.01), (10.0, 0.0, 1.0), (0.0, 10.0, 1.0)),
        dtype=wp.vec3,
        device=self.device,
    )
    constraint.prepare(crossed)
    contacts = constraint.cloth_vertex_face_contacts

    self.assertGreaterEqual(int(contacts.count.numpy()[0]), 1)
    np.testing.assert_allclose(contacts.directions.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-6)
    self.assertGreater(float(contacts.depths.numpy()[0]), constraint.thickness)
```

Add a one-step `SolverLIMX` integration test with a uniformly falling cloth triangle and assert its vertices do not finish below the rigid face.

Extend `_make_vf_fixture` with exact defaults `cloth_velocities=None`,
`dt=1.0 / 600.0`, and `prepare=True`. When velocities are omitted, create
three zero vectors; otherwise pass the supplied values to
`ModelBuilder.add_particles`. Always call `begin_step(..., dt)`, and call
`prepare()` only when `prepare` is true.

- [ ] **Step 2: Run the VF tests and verify the expected failure**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py -k "step_start_side or high_speed"
```

Expected: FAIL because the discrete detector emits no contact after the 10 mm crossing, or flips the direction to negative Z.

- [ ] **Step 3: Add step caches and swept bounds**

In `ConstraintKinematicMeshContact.__init__`, allocate:

```python
self._step_positions = wp.clone(model.particle_q)
self._predicted_positions = wp.clone(model.particle_q)
self._step_collider_positions = wp.clone(self.collider_positions)
self._predicted_collider_positions = wp.clone(self.collider_positions)
```

Add a Warp prediction kernel with this exact behavior:

```python
@wp.kernel
def predict_positions(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    dt: float,
    predicted: wp.array[wp.vec3],
):
    index = wp.tid()
    predicted[index] = positions[index] + dt * velocities[index]
```

In `begin_step()`, copy current cloth/collider positions and launch the kernel for both surfaces. Replace current-only triangle and edge bounds with swept bounds containing the step-start, predicted, and current-iterate endpoints. Refit all four BVHs from those bounds in `prepare()`.

- [ ] **Step 4: Make both VF kernels use a preserved side**

Pass step-start and predicted positions into both VF detection kernels. Orient the current face normal to the step-start face normal, choose the outward sign from the step-start signed distance, and compute:

```python
signed_gap = wp.dot(current_relative_position, oriented_normal)
depth = thickness - signed_gap
```

Keep a candidate when either the current signed gap or linearly predicted signed gap is below `thickness`, and accept an interior projection at either the current or predicted endpoint. Store the preserved oriented normal and signed depth, including negative depth for predictive candidates outside the active zone.

- [ ] **Step 5: Gate inactive normal force and Hessian consistently**

At the start of `accumulate_contact_force`, `contact_hessian_multiply`, and `accumulate_contact_diagonal`, return when `depths[contact] <= 0.0`. Keep approaching damping available for predictive candidates. Friction already returns when its normal load is nonpositive.

- [ ] **Step 6: Run focused and existing VF/operator tests**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py
```

Expected: all tests pass, including the high-speed solver integration and existing dense force/HVP/diagonal checks.

- [ ] **Step 7: Commit the VF fix**

```bash
git add newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py \
  newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py \
  newton/tests/test_solver_limx_kinematic_mesh_contact.py
git commit -m "Prevent LIMX kinematic VF crossing"
```

---

### Task 2: Preserve EE orientation and second-order consistency

**Files:**
- Modify: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py`
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`

**Interfaces:**
- Consumes: Task 1 step-start/predicted arrays and swept edge BVHs.
- Produces: one-sided predictive cloth-E/rigid-E contacts whose mollified normal force, HVP, diagonal, damping, and friction share one active state.

- [ ] **Step 1: Write the failing EE crossing and inactive-operator tests**

Add a crossed-edge test with a cloth edge starting at Z=10 mm, velocity Z=-2 m/s, a perpendicular rigid edge at Z=0, and a current iterate at Z=-10 mm:

```python
def test_keeps_edge_contact_on_step_start_side_after_crossing(self):
    """Keep a swept cloth edge on its step-start side of a rigid edge."""
    constraint, _state = self._make_vf_fixture(
        cloth_positions=((-0.5, 0.0, 0.01), (0.5, 0.0, 0.01), (-0.5, 0.0, 1.0)),
        rigid_vertices=((0.0, -0.5, 0.0), (0.0, 0.5, 0.0), (0.0, -0.5, -1.0)),
        cloth_velocities=((0.0, 0.0, -2.0), (0.0, 0.0, -2.0), (0.0, 0.0, 0.0)),
        dt=0.01,
        prepare=False,
    )
    crossed = wp.array(
        ((-0.5, 0.0, -0.01), (0.5, 0.0, -0.01), (-0.5, 0.0, 1.0)),
        dtype=wp.vec3,
        device=self.device,
    )
    constraint.prepare(crossed)
    contacts = constraint.edge_edge_contacts
    self.assertGreaterEqual(int(contacts.count.numpy()[0]), 1)
    self.assertGreater(float(contacts.depths.numpy()[0]), constraint.thickness)
    np.testing.assert_allclose(contacts.directions.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-6)
```

Add a `_KinematicEdgeEdgeContactBuffer` test with depth `-0.01` and assert normal force, HVP, and diagonal are exactly zero while the buffer remains populated.

- [ ] **Step 2: Run the EE tests and verify the expected failure**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py -k "edge_contact_on_step_start_side or inactive_predictive"
```

Expected: FAIL because the current EE detector drops the crossed pair and mollified normal paths do not gate negative depth.

- [ ] **Step 3: Add predictive one-sided EE detection**

Pass cloth/collider step-start and predicted endpoints to `detect_cloth_edge_rigid_edge`. Use the current closest-point parameters when interior, otherwise use predicted closest-point parameters when those are interior. Derive a reference separation from step-start endpoints at the selected parameters. Orient the current separation direction to that reference and preserve its sign:

```text
current_direction = normalize(current_separation)
if dot(current_direction, reference_direction) < 0:
    current_direction = -current_direction
signed_gap = dot(current_separation, current_direction)
depth = thickness - signed_gap
```

Retain the candidate when current or predicted signed gap enters the thickness zone. Continue storing current rigid edge vectors and the existing rest-edge mollifier threshold.

- [ ] **Step 4: Gate mollified normal paths with the same active depth**

Return early for `depths[contact] <= 0.0` in:

- `accumulate_mollified_edge_edge_force`;
- `mollified_edge_edge_hessian_multiply`;
- `accumulate_mollified_edge_edge_diagonal`.

Leave predictive damping enabled and leave friction disabled through its nonpositive normal-load check.

- [ ] **Step 5: Run all kinematic-contact tests**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py
```

Expected: all VF, EE, mollifier, friction, damping, and dense-reference tests pass.

- [ ] **Step 6: Commit the EE fix**

```bash
git add newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py \
  newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py \
  newton/tests/test_solver_limx_kinematic_mesh_contact.py
git commit -m "Prevent LIMX kinematic EE crossing"
```

---

### Task 3: Activate the Franka box-grasp scene

**Files:**
- Modify: `newton/tests/test_example_cloth_limx_franka.py`
- Modify: `newton/examples/cloth/example_cloth_limx_franka.py`
- Modify: `CHANGELOG.md`
- Modify: `docs/images/examples/example_cloth_limx_franka.jpg`

**Interfaces:**
- Consumes: predictive `ConstraintKinematicMeshContact` from Tasks 1 and 2, existing Franka IK targets, `ConstraintSelfCollision`, `ConstraintTriangleElastic`, and `ConstraintDihedralBending`.
- Produces: an active physical `cloth_limx_franka` example with `test_final()` lift metrics and a refreshed screenshot.

- [ ] **Step 1: Write failing active-cloth and lift tests**

Change the initial-scene test to require every cloth particle to be active. Replace the old fixed-position final assertion with physical behavior:

```python
def test_grasp_sequence_lifts_active_cloth_with_box_contacts(self):
    """Lift active cloth using kinematic Franka box contacts and friction."""
    frame_count = int(np.ceil(module.SEQUENCE_DURATION * module.FPS)) + 1
    example = module.Example(ViewerNull(num_frames=frame_count), SimpleNamespace(graph_capture=True))
    for _ in range(frame_count):
        example.step()
        example.test_post_step()
    example.test_final()

    self.assertEqual(example.sim_substeps, 1)
    self.assertAlmostEqual(example.sim_dt, 0.01)
    self.assertGreater(example.maximum_cloth_lift, 0.10)
    self.assertGreaterEqual(example.captured_hold_duration, 0.5)
    self.assertEqual(example.maximum_overflow_count, 0)
    self.assertEqual(example.maximum_box_intersection_count, 0)
```

- [ ] **Step 2: Run the example tests and verify the expected failure**

Run:

```bash
uv run newton/tests/test_example_cloth_limx_franka.py
```

Expected: FAIL because the current example marks all cloth particles inactive and never changes their height.

- [ ] **Step 3: Replace fixed cloth and Featherstone stepping with LIMX cloth**

Set `FPS = 100` and `SIM_SUBSTEPS = 1`. Keep particles active. Construct the same triangle elasticity, dihedral bending, adaptive self-collision, and LIMX solver parameters used by `example_cloth_limx_franka_drop.py`.

Select the table plus Franka wrist, hand, and finger collision shapes by collision flag and label. Pass those box/mesh collision proxies to `ConstraintKinematicMeshContact` with thickness 3 mm, stiffness 20 kN/m, damping 0.5 N·s/m, friction 0.4, and capacity 65,536.

- [ ] **Step 4: Drive rigid state kinematically before each cloth step**

After IK writes `target_joint_q`, compute target joint velocities, assign target joint coordinates and velocities to `state_0`, and call:

```python
newton.eval_fk(
    self.model,
    self.state_0.joint_q,
    self.state_0.joint_qd,
    self.state_0,
)
self.kinematic_contact.update_colliders(self.state_0.body_q, self.state_0.body_qd)
self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
self.state_0.assign(self.state_1)
```

Remove `SolverFeatherstone` and the temporary mutation of `model.particle_count` and gravity. Keep the entire path CUDA-graph capturable.

- [ ] **Step 5: Add close, lift, and raised-hold metrics**

Extend the keyframes with at least 0.5 s closed at the raised pose before reopening. Record pre-lift centroid height when closure completes, maximum later centroid height, consecutive raised-and-captured duration, active contact counts, overflow, and collision-surface triangle intersections. Implement `_count_surface_intersections()` with segment-triangle tests in both directions between cloth and collider surfaces; call it from `test_post_step()` only after finger closure so interactive rendering does not pay for CPU geometric diagnostics. `test_final()` must reject nonfinite state, no contact, less than 0.10 m centroid lift, less than 0.5 s raised capture, any box intersection, or any overflow.

Expose collision proxies in the viewer so the box contact surface is visually distinguishable from the larger finger render mesh.

- [ ] **Step 6: Run the full headless grasp rollout**

Run:

```bash
uv run newton/tests/test_example_cloth_limx_franka.py
```

Expected: both initial-state and full grasp tests pass at 100 Hz with one step per frame.

- [ ] **Step 7: Diagnose one parameter at a time if the pinch fails**

Use the recorded box intersections, contact counts, centroid trajectory, and relative slip to identify the limiting term. Correct predictive contact before tuning physical parameters. Once geometry remains separated, change only friction if the cloth slides; keep thickness at 3 mm and do not add substeps.

- [ ] **Step 8: Update the changelog and screenshot**

Insert entries at nonterminal positions in `[Unreleased]`:

```markdown
- Make the LIMX Franka scene physically grasp active cloth with kinematic box contact and friction.
- Prevent fast LIMX kinematic VF/EE candidates from flipping sides after crossing a collider surface.
```

Run the example with the project screenshot workflow and replace `docs/images/examples/example_cloth_limx_franka.jpg` with a 320-by-320 frame showing the cloth held above the table.

- [ ] **Step 9: Run focused verification and pre-commit**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py
uv run newton/tests/test_example_cloth_limx_franka.py
uv run newton/tests/test_example_cloth_limx_franka_drop.py
uvx pre-commit run -a
```

Expected: every command succeeds without warnings or file rewrites on the final pre-commit run.

- [ ] **Step 10: Commit the physical grasp scene**

```bash
git add CHANGELOG.md \
  docs/images/examples/example_cloth_limx_franka.jpg \
  newton/examples/cloth/example_cloth_limx_franka.py \
  newton/tests/test_example_cloth_limx_franka.py
git commit -m "Make LIMX Franka grasp active cloth"
```

- [ ] **Step 11: Run the interactive acceptance scene**

Run:

```bash
uv run -m newton.examples cloth_limx_franka --viewer gl --no-headless \
  --num-frames 630 --render-fps 100
```

Acceptance: the collision boxes close around the cloth, the cloth rises at least 0.10 m, and it remains held for at least 0.5 s without box-surface intersection.
