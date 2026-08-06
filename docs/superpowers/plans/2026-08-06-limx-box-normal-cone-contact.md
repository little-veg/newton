# LIMX Box Normal-Cone Contact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter box reverse-VF and EE candidates by rigid-feature outward normal cones so an opened Franka gripper releases cloth without losing valid exterior edge and vertex contact.

**Architecture:** Generate box vertex and edge normal-cone metadata beside the existing triangulated collider topology, rotate it with the kinematic body pose, and use it only during contact candidate ownership tests. Accepted stencils keep the existing LIMX force, damping, friction, HVP, and diagonal-Hessian paths unchanged; arbitrary mesh colliders remain two-sided.

**Tech Stack:** Python 3, NumPy, Warp CUDA kernels, Newton LIMX, `unittest`, `uv`.

## Global Constraints

- Keep `dt = 0.01 s`, `SIM_SUBSTEPS = 1`, and collision thickness `0.003 m`.
- Do not add dependencies or change the public `ConstraintKinematicMeshContact` constructor.
- Preserve mesh and convex-mesh two-sided behavior.
- Every changed test method uses a triple-double-quoted imperative docstring.
- Keep force, damping, friction, HVP, and diagonal Hessian consistent for every accepted contact.

---

### Task 1: Lock the failure down with real contact and release tests

**Files:**
- Modify: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`
- Modify: `newton/tests/test_example_cloth_limx_franka.py`

**Interfaces:**
- Consumes: `ConstraintKinematicMeshContact.prepare(positions)` and its three real frozen-contact buffers.
- Produces: regression expectations for invalid inward-side box features, valid exterior box features, and post-open Franka release.

- [ ] **Step 1: Add reverse-VF ownership cases**

Create a real box and one horizontal cloth triangle near its bottom vertices. Add one test with the triangle just above the bottom plane, where bottom-vertex-to-cloth separation points into the box and reverse VF must be absent. Add a second case just below the bottom plane and assert reverse VF remains present.

```python
def _make_box_bottom_fixture(self, cloth_z: float):
    builder = newton.ModelBuilder()
    body = builder.add_body()
    shape = builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    positions = ((0.05, 0.05, cloth_z), (0.15, 0.05, cloth_z), (0.1, 0.15, cloth_z))
    builder.add_particles(
        pos=[wp.vec3(*position) for position in positions],
        vel=[wp.vec3(0.0)] * 3,
        mass=[1.0] * 3,
        radius=[0.003] * 3,
    )
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=self.device)
    state = model.state()
    constraint = newton.solvers.ConstraintKinematicMeshContact(
        model, [shape], 0.003, 2.0e4, 0.5, 0.0, 1.0e-2, max_contacts=64
    )
    constraint.update_colliders(state.body_q, state.body_qd)
    constraint.begin_step(state.particle_q, state.particle_qd, 0.01)
    return constraint, state

def test_rejects_box_bottom_vertices_from_their_inward_side(self):
    """Reject reverse VF contacts viewed through a box vertex's inward side."""
    constraint, state = self._make_box_bottom_fixture(cloth_z=-0.099)
    constraint.prepare(state.particle_q)
    self.assertEqual(int(constraint.rigid_vertex_face_contacts.count.numpy()[0]), 0)

def test_keeps_box_bottom_vertices_from_their_outward_side(self):
    """Keep reverse VF contacts in a box vertex's outward normal cone."""
    constraint, state = self._make_box_bottom_fixture(cloth_z=-0.101)
    constraint.prepare(state.particle_q)
    self.assertGreater(int(constraint.rigid_vertex_face_contacts.count.numpy()[0]), 0)
```

- [ ] **Step 2: Add EE ownership cases**

Use a cloth edge crossing the projection of a physical bottom box edge. The edge just above the bottom plane is outside the rigid edge cone and must be rejected; the same edge just below the box is a valid exterior EE candidate.

```python
def _make_box_edge_fixture(self, cloth_z: float):
    builder = newton.ModelBuilder()
    body = builder.add_body()
    shape = builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    positions = ((0.0, -0.15, cloth_z), (0.0, -0.05, cloth_z), (0.001, -0.15, cloth_z))
    builder.add_particles(
        pos=[wp.vec3(*position) for position in positions],
        vel=[wp.vec3(0.0)] * 3,
        mass=[1.0] * 3,
        radius=[0.003] * 3,
    )
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=self.device)
    state = model.state()
    constraint = newton.solvers.ConstraintKinematicMeshContact(
        model, [shape], 0.003, 2.0e4, 0.5, 0.0, 1.0e-2, max_contacts=64
    )
    constraint.update_colliders(state.body_q, state.body_qd)
    constraint.begin_step(state.particle_q, state.particle_qd, 0.01)
    return constraint, state

def _target_edge_contact_count(self, constraint) -> int:
    contacts = constraint.edge_edge_contacts
    count = min(int(contacts.count.numpy()[0]), contacts.capacity)
    ids = np.sort(contacts.ids.numpy()[:count], axis=1)
    return int(np.count_nonzero(np.all(ids == (0, 1), axis=1)))

def test_rejects_box_bottom_edge_from_its_inward_side(self):
    """Reject EE contacts outside a box edge's outward normal cone."""
    constraint, state = self._make_box_edge_fixture(cloth_z=-0.099)
    constraint.prepare(state.particle_q)
    self.assertEqual(self._target_edge_contact_count(constraint), 0)

def test_keeps_box_bottom_edge_from_its_outward_side(self):
    """Keep EE contacts inside a box edge's outward normal cone."""
    constraint, state = self._make_box_edge_fixture(cloth_z=-0.101)
    constraint.prepare(state.particle_q)
    self.assertGreater(self._target_edge_contact_count(constraint), 0)
```

- [ ] **Step 3: Extend the Franka rollout through release**

Run 160 frames after the existing 6.4 s trajectory reaches fully open fingers. Preserve the lift/hold assertions, then require the formerly pinned raised patch to fall at least 5 cm below the finger-box bottom and require no high cloth contact to remain within the gripper collision boxes.

```python
release_frames = int(1.6 * module.FPS)
for _ in range(frame_count + release_frames):
    example.step()

finger_bottom = example.kinematic_contact.collider_positions.numpy()[8:, 2].min()
self.assertLess(example.state_0.particle_q.numpy()[:, 2].max(), finger_bottom - 0.05)
```

- [ ] **Step 4: Run the RED tests**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py
uv run newton/tests/test_example_cloth_limx_franka.py
```

Expected: the new inward-side primitive tests and release assertion fail against the unfiltered implementation; existing exterior, lift, hold, and non-penetration assertions pass.

---

### Task 2: Generate and rotate box feature normal cones

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`
- Test: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`

**Interfaces:**
- Produces: `collider_vertex_cone_normals: wp.array2d[wp.vec3]`, `collider_vertex_cone_counts: wp.array[int]`, `collider_edge_cone_normals: wp.array2d[wp.vec3]`, and `collider_edge_cone_counts: wp.array[int]` in current world orientation.
- Consumes: box topology from `_box_surface()`, `MeshAdjacency.edge_indices`, shape-local transforms, and current rigid `body_q`.

- [ ] **Step 1: Add local vector rotation and box vertex metadata**

Add `_transform_vectors()` beside `_transform_points()`. For each of the eight box vertices, generate the three signed local basis normals corresponding to its incident faces, rotate them through `model.shape_transform`, and store count `3`. Store three zero vectors and count `0` for mesh vertices.

```python
def _transform_vectors(vectors: np.ndarray, transform: np.ndarray) -> np.ndarray:
    quaternion = transform[3:]
    vector = quaternion[:3]
    scalar = quaternion[3]
    twice_cross = 2.0 * np.cross(vector, vectors)
    return vectors + scalar * twice_cross + np.cross(vector, twice_cross)
```

- [ ] **Step 2: Derive box edge cones from shared vertex normals**

For each adjacency edge, intersect the endpoint cone normals. A physical box edge gets two shared normals, a face diagonal gets one, and a mesh edge gets zero. Store three slots per feature so the same Warp helper handles vertices and edges.

```python
def _edge_normal_cones(
    edges: np.ndarray,
    vertex_normals: np.ndarray,
    vertex_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros((len(edges), 3, 3), dtype=np.float32)
    counts = np.zeros(len(edges), dtype=np.int32)
    # Copy each outward normal shared by both edge endpoints once.
    return normals, counts
```

- [ ] **Step 3: Rotate cone normals with each body**

Add `_update_collider_feature_normals`, using `wp.transform_get_rotation(body_q[body])` and `wp.quat_rotate`. Launch it for vertex and edge cone arrays from `update_colliders()` immediately after updating collider vertices.

Derive `collider_edge_body` from `bodies_np[edges_np[:, 2]]` and assert that
each edge's second endpoint has the same body before constructing the Warp
array.

```python
@wp.kernel
def _update_collider_feature_normals(local_normals, body_indices, body_q, world_normals):
    feature, slot = wp.tid()
    body = body_indices[feature]
    normal = local_normals[feature, slot]
    world_normals[feature, slot] = normal if body < 0 else wp.quat_rotate(
        wp.transform_get_rotation(body_q[body]), normal
    )
```

- [ ] **Step 4: Run topology and transform tests**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py
```

Expected: existing topology/transform tests pass; inward-side behavior remains RED until Task 3.

---

### Task 3: Filter reverse VF and EE by normal-cone ownership

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py`
- Modify: `newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py`
- Test: `newton/tests/test_solver_limx_kinematic_mesh_contact.py`

**Interfaces:**
- Consumes: the four world-space cone arrays from Task 2.
- Produces: `_inside_feature_normal_cone()` and ownership-filtered `detect_rigid_vertex_cloth_face()` / `detect_cloth_edge_rigid_edge()` contact buffers.

- [ ] **Step 1: Add the Warp cone-membership helper**

Project the rigid-to-cloth separation onto the feature's orthonormal outward normals. Reject a negative cone coefficient or a residual outside their span. Count zero returns true for legacy mesh behavior.

```python
@wp.func
def _inside_feature_normal_cone(separation, normals, feature, normal_count):
    if normal_count == 0:
        return True
    projection = wp.vec3(0.0)
    tolerance = wp.max(_MIN_CONTACT_DISTANCE, 1.0e-3 * wp.length(separation))
    for slot in range(3):
        if slot < normal_count:
            coefficient = wp.dot(separation, normals[feature, slot])
            if coefficient < -tolerance:
                return False
            projection += coefficient * normals[feature, slot]
    return wp.length(separation - projection) <= tolerance
```

- [ ] **Step 2: Filter rigid-vertex/cloth-face candidates**

Pass vertex cone normals/counts into `detect_rigid_vertex_cloth_face`. Test `projected - rigid_position` for the current pair and `predicted_projected - rigid_predicted_position` for the predicted pair. Continue only when an interior projection also belongs to the rigid vertex cone.

- [ ] **Step 3: Filter cloth-edge/rigid-edge candidates**

Pass edge cone normals/counts into `detect_cloth_edge_rigid_edge`. Evaluate current and predicted closest-pair separations before selecting contact parameters. Preserve the existing swept reference direction, mollifier, and frozen Hessian data for accepted contacts.

- [ ] **Step 4: Run the primitive GREEN tests**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py
```

Expected: all inward-side rejection and valid exterior preservation tests pass, along with the existing force/HVP/diagonal tests.

- [ ] **Step 5: Commit the core correction**

```bash
git add newton/_src/solvers/limx/constraints/kinematic_mesh_contact.py \
        newton/_src/solvers/limx/constraints/kinematic_mesh_contact_kernels.py \
        newton/tests/test_solver_limx_kinematic_mesh_contact.py
git commit -m "Filter LIMX box contacts by normal cone"
```

---

### Task 4: Verify release and close out the user-facing correction

**Files:**
- Modify: `newton/tests/test_example_cloth_limx_franka.py`
- Modify: `CHANGELOG.md`
- Modify if the accepted render changes: `docs/images/examples/example_cloth_limx_franka.jpg`

**Interfaces:**
- Consumes: ownership-filtered contact from Task 3 and the existing Franka keyframes.
- Produces: a regression-protected grasp/hold/release rollout and a visible GL acceptance run.

- [ ] **Step 1: Run the Franka release test**

Run:

```bash
uv run newton/tests/test_example_cloth_limx_franka.py
```

Expected: the cloth still lifts more than 0.10 m, holds at least 0.5 s, has zero vertices inside finger boxes, and falls away after the fingers open. If release remains locked, inspect retained feature cones and correct ownership; do not tune friction, thickness, stiffness, or substeps.

- [ ] **Step 2: Add the Unreleased Fixed changelog entry**

Add at a nonterminal position in `CHANGELOG.md`'s Unreleased `Fixed` category:

```markdown
- Release LIMX cloth from kinematic box edges by filtering contacts to rigid outward normal cones.
```

- [ ] **Step 3: Run focused and registered verification**

Run:

```bash
uv run newton/tests/test_solver_limx_kinematic_mesh_contact.py
uv run newton/tests/test_example_cloth_limx_franka.py
uv run newton/tests/test_example_cloth_limx_franka_drop.py
uv run --extra dev -m newton.tests -k cloth_limx_franka
uvx pre-commit run -a
```

Expected: every command exits `0`; no contact buffer overflows or non-finite state are reported.

- [ ] **Step 4: Commit behavior and documentation**

```bash
git add newton/tests/test_example_cloth_limx_franka.py CHANGELOG.md \
        docs/images/examples/example_cloth_limx_franka.jpg
git commit -m "Verify LIMX Franka cloth release"
```

- [ ] **Step 5: Launch interactive acceptance**

Run:

```bash
uv run -m newton.examples cloth_limx_franka --viewer gl --no-headless \
    --num-frames 100000 --render-fps 100
```

Expected: the gripper pinches and lifts the cloth, opens, and the cloth visibly releases without tunneling through the finger boxes.
