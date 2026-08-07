# LIMX Cloth Table-Fold Design

## Goal

Add an interactive LIMX example that starts with a `100 x 100` cloth grid lying flat on a table and dynamically folds the right half over the left half.

## Scene Geometry

The cloth is a `1 m x 1 m`, `0.3 kg` square with 100 cells per side, 10,201 particles, and 20,000 alternating-diagonal triangles. It starts horizontally just above a `1.3 m x 1.3 m` box table. Gravity remains enabled.

The cloth uses a 3 mm particle radius and 3 mm collision thickness. Its left boundary is anchored at its initial table-supported position so the low-friction table does not let the entire sheet translate during folding. The right boundary is driven; interior particles, the fold line, and both cloth halves remain simulated.

## Fold Motion

The right boundary follows a smooth half-circle about the cloth's center fold line. A smoothstep phase drives the angle from 0 to 180 degrees over 4 seconds. The boundary begins at the right table edge, rises above the cloth at mid-fold, and finishes over the left edge.

At the final pose, the driven boundary remains 6 mm above the lower layer. This represents the two-layer separation and prevents the anchor constraint from demanding coincident cloth surfaces. The example holds the folded state for 2 seconds so contact stability and chatter are visible. It does not release the driven edge.

## Constraints and Solver

Static constraints consist of `1.0e7 N/m` left and right boundary anchors, triangle elasticity with per-triangle stiffness `(1.0e4, 1.0e4, 1.0e3) N/m`, and dihedral bending with stiffness `1.0e-5`. Anchor targets for the right boundary are updated before every simulation step.

Dynamic constraints combine cloth self-collision and table contact:

- Self-collision uses the existing LIMX VF/EE implementation, 3 mm thickness, adaptive stiffness, friction `0.3`, and capacity for 262,144 contacts.
- Table contact uses a table-only `ConstraintKinematicMeshContact`, 3 mm thickness, stiffness `2.0e4`, friction `0.05`, zero normal damping, and no CCD because the table is stationary.

The solver uses one `0.01 s` step per rendered frame with no substeps and no added damping intended to hide collision instability. No public solver API changes or new dependencies are required.

## Example and Validation

The example module is `newton/examples/cloth/example_cloth_limx_fold.py` and runs as:

```bash
uv run -m newton.examples cloth_limx_fold
```

A focused `unittest` verifies the 100-cell resolution, particle and triangle counts, boundary ownership, and right-edge targets at 0, 90, and 180 degrees. A short CUDA smoke rollout verifies finite state and table support without adding a large test matrix. The final acceptance step is an interactive run showing the cloth supported by the table throughout the fold, the upper layer ending over the lower layer, and no obvious sustained penetration or chatter during the hold.

## Non-goals

- Using a robot or rigid gripper to perform the fold.
- Releasing the folded edge after the hold.
- Changing the LIMX contact APIs, collision thickness model, or damping formulation.
- Adding substeps or velocity damping as a stability workaround.
