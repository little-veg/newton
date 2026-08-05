# LIMX Kinematic Mesh Contact Design

## Goal

Add matrix-free vertex-face (VF) and edge-edge (EE) contact between LIMX cloth
and a triangulated kinematic rigid surface. The first example drops a free
square cloth onto a stationary Franka wrist and open gripper. The cloth must
remain supported without a table, avoid visible interpenetration, and settle
without persistent jitter.

This contact path deliberately uses surface stencils rather than Newton's SDF
rigid-soft collision path. It reuses the Warp VF/EE structure already present
in Newton VBD and LIMX self-collision while keeping the rigid surface outside
the LIMX unknown vector.

## Scope

The first version includes:

- two-sided cloth-vertex/rigid-face contacts;
- two-sided rigid-vertex/cloth-face contacts;
- cloth-edge/rigid-edge contacts with the IPC near-parallel EE mollifier;
- normal penalty, approaching-only normal damping, and regularized Coulomb
  friction;
- force, matrix-free Hessian-vector products, and exact block-Jacobi diagonal
  blocks for every contact term;
- triangle-surface extraction for selected Newton mesh and box shapes;
- kinematic collider transforms, with zero velocity in the first example;
- fixed-capacity, CUDA-capture-safe contact buffers with observable overflow;
- a visual drop-and-settle Franka example and focused operator checks.

The first version does not include continuous collision detection, adaptive
collision thickness, rigid-body reaction wrenches, two-way coupling, or
support for every analytic Newton shape type. These can be added without
changing the cloth-side operator contract.

## Public Constraint

Expose a new dynamic constraint as
`newton.solvers.ConstraintKinematicMeshContact`. Its constructor accepts the
particle-triangle `Model`, selected rigid `shape_indices`, fixed contact
parameters, capacities, and device. The initial shape extractor accepts mesh
shapes directly and triangulates each box into twelve triangles. Unsupported
selected geometry raises a clear error rather than silently changing the
collision representation.

The constraint provides the normal LIMX dynamic-operator methods:

```python
begin_step(positions, velocities, dt)
prepare(positions)
accumulate_force(positions, output)
hessian_multiply(positions, vector, output)
accumulate_diagonal(positions, output)
```

It additionally provides an explicit collider update method accepting current
rigid body transforms and velocities. The stationary example calls this once
after initializing FK. A future moving-arm example can call it before each
LIMX substep without changing the contact or Hessian interface.

The constraint composes with `ConstraintSelfCollision` through
`ConstraintGroupDynamic`; it does not replace cloth self-collision.

## Kinematic Surface Representation

At construction, flatten the selected rigid shapes into one collider surface.
Each collider vertex stores its body index, body-local position including the
shape transform, and a stable surface vertex index. Store triangle indices and
derive unique edges with `MeshAdjacency`. Shapes remain topologically separate,
so no edge is introduced between distinct collision shapes.

Collider world positions and velocities are device arrays. A Warp kernel
updates them from body transforms and spatial velocities. The stationary
example supplies zero collider velocity. Triangle and edge AABBs are updated
and their fixed-topology BVHs refitted during `prepare()`.

The cloth already supplies dynamic vertex positions, triangles, edges, and
BVHs. Cross-mesh detection needs no self-topology exclusion because no cloth
primitive shares an index with a collider primitive.

## Contact Detection

Rebuild contacts at every LIMX nonlinear iteration, then freeze the selected
feature parameters, normals, and depths for force and PCG operations in that
iteration.

Generate all three required cross-surface stencil families:

1. Cloth vertex against rigid face. Query the rigid triangle BVH from every
   cloth vertex, project onto each candidate face, and keep interior face
   projections within the activation distance.
2. Rigid vertex against cloth face. Query the cloth triangle BVH from every
   rigid vertex and retain interior projections. This direction prevents
   rigid corners from passing through triangle interiors even when no cloth
   vertex is nearby.
3. Cloth edge against rigid edge. Query the rigid edge BVH from every cloth
   edge and retain interior/interior closest points within the activation
   distance.

VF is two-sided: the frozen direction follows the signed face-plane side at
detection. Exact zero-distance and degenerate configurations are skipped
rather than emitting a nonfinite normal. EE contacts retain the current IPC
near-parallel mollifier, with its derivatives restricted to the dynamic cloth
endpoints.

The first version uses discrete detection. The example uses ten substeps per
60 Hz frame, so the expected impact displacement stays below the 3 mm
activation distance. CCD is required later for faster robot or cloth motion.

## Frozen Second-Order Contact Model

For an active contact, collect only dynamic cloth vertices in `x`. Frozen
weights `q_i`, direction `n`, and rigid constant `c` define

```text
g(x) = n dot (sum_i(q_i x_i) + c)
r(x) = h - g(x)
E_n(x) = 1/2 k r(x)^2.
```

The rigid constant contains the frozen rigid face point, vertex, or edge point.
The dynamic stencil arities are one for cloth-V/rigid-F, three for
rigid-V/cloth-F, and two for cloth-E/rigid-E.

The physical normal force and positive-semidefinite frozen-contact Hessian are

```text
f_i = k r q_i n
(H v)_i = k q_i n sum_j(q_j dot(n, v_j))
H_ii = k q_i^2 outer(n, n).
```

The operator implements all three expressions. It must never add a rigid
vertex to the LIMX unknown vector or drop the off-diagonal cloth coupling from
the HVP. The diagonal method supplies the exact diagonal blocks of the same
frozen Hessian used by PCG.

Approaching-only normal damping uses frozen relative normal velocity. Its
position derivative through the step velocity approximation contributes the
same rank-one structure scaled by `normal_damping / dt`. Regularized Coulomb
friction follows the existing LIMX self-collision tangent model and contributes
its own force, HVP, and diagonal blocks. Static collider motion is zero in the
first example, but the stored rigid surface velocity keeps the relative-motion
definition valid for a later moving collider.

For near-parallel EE contacts, reuse the IPC-mollified residual already used by
`ConstraintSelfCollision`. The rigid edge is constant; the residual Jacobian
and Gauss-Newton HVP differentiate only the cloth edge endpoints. Normal and
friction contributions must use the same mollifier activation state.

## Solver Data Flow

For every simulation substep:

1. Update or confirm the collider transforms and surface velocities.
2. Call `begin_step()` to cache cloth step-start velocity and friction anchors.
3. At each nonlinear iteration, call `prepare()` to refit both surface BVHs,
   detect VF/EE contacts, and cache frozen force/Hessian data.
4. Accumulate normal, damping, and friction forces into the LIMX right-hand
   side.
5. Add every contact's diagonal Hessian blocks before inverting the block-Jacobi
   preconditioner.
6. Evaluate the same frozen normal, damping, friction, and mollified EE Hessians
   in every PCG matrix-vector product.

Contact detection never runs inside PCG. All counters and overflow counters
remain on the device so the path remains graph-capture compatible.

## Franka Drop Example

Create `cloth/example_cloth_limx_franka_drop.py` as the first interaction
scene, reusing geometry helpers from `cloth/example_cloth_limx_franka.py`.
Keep the existing fixed-cloth grasp scaffold unchanged so later moving-arm
work retains its visual baseline. The drop scene will:

- keep the Franka fixed in a pose with the wrist and open gripper facing up;
- select the wrist, hand, and finger collision shapes for the kinematic surface;
- activate every vertex of a 0.4 m, 21-by-21 square cloth;
- release the horizontal cloth about 0.1 m above the hand/wrist target;
- remove the table and ground contact so a sliding or tunneling cloth cannot
  appear successful after landing elsewhere;
- retain triangle elasticity, dihedral bending, and LIMX cloth self-collision;
- use 60 Hz rendering with ten simulation substeps;
- start with 3 mm rigid-cloth activation distance, 20 kN/m fixed normal
  stiffness, 0.5 N·s/m approaching-only normal damping, friction coefficient
  0.4, a 0.01 m/s friction regularization threshold, and a capacity of 65,536
  contacts for each stencil family;
- use two nonlinear iterations and 50 PCG iterations per substep.

Render the cloth and rigid collision surface clearly enough to inspect contact.
The run lasts about six seconds. A successful result remains near the
hand/wrist, shows no visible surface crossing, and maintains cloth RMS speed
below approximately 0.02 m/s over the final second.

## Failure Handling and Diagnostics

Constructor validation rejects invalid shape indices, unsupported selected
geometry, malformed topology, invalid parameters, or device mismatches.
Runtime narrow phase skips degenerate primitives and records contact buffer
overflow explicitly.

The example reports attempted and retained counts for both VF directions and
EE, maximum detected penetration, final-second RMS speed, and collider support
status. If the first run does not settle, preserve and show that visual result
before tuning. Use those diagnostics to decide whether stiffness, damping,
friction, substeps, or the discrete detector is limiting stability.

## Focused Validation

Use `unittest`. Add small synthetic stencils that check:

- cloth-V/rigid-F, rigid-V/cloth-F, and cloth-E/rigid-E candidate generation;
- correct dynamic cloth ids, weights, directions, and depths;
- equality of matrix-free HVP with an independently assembled dense frozen
  Hessian, including its off-diagonal cloth blocks;
- exact block diagonal and nonnegative quadratic forms;
- normal damping and friction contributions in force, HVP, and diagonal paths;
- dynamic-only differentiation of the IPC EE mollifier;
- finite empty/degenerate behavior and explicit overflow accounting.

Run the focused operator tests and one headless example rollout. The primary
acceptance check is the rendered six-second drop. Do not run the complete
Newton test suite for this visual iteration unless integration work later
requires release-level verification.
