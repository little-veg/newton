# LIMX Franka Split-Friction Design

## Goal

Let the cloth slide freely on the table without weakening the gripper's ability to hold it. The table uses a rigid-cloth friction coefficient of `0.05`; the two finger pads retain `0.4`.

## Design

The Franka example will create two `ConstraintKinematicMeshContact` operators from the existing collider selection:

- `table_contact` owns only the table box, uses friction `0.05`, and disables CCD because the table is stationary.
- `gripper_contact` owns both finger-pad boxes, uses friction `0.4`, and keeps CCD enabled for moving-rigid contact.

Both operators keep the current 3 mm thickness, `2.0e4` normal stiffness, zero normal damping, contact capacity, and friction regularization. The dynamic constraint group evaluates self-collision, table contact, and gripper contact in that order. Each simulation step updates both collider operators from the same rigid-body state before the LIMX solve.

This split is local to the example. It does not expand `ConstraintKinematicMeshContact` with per-shape material arrays, avoiding a broader solver API change for one scene-specific material distinction.

## Metrics and Validation

Contact and overflow metrics will aggregate the table and gripper operators. CCD metrics will come only from `gripper_contact`. Gripper geometry and penetration checks will read the gripper operator directly rather than relying on fixed offsets in one combined collider buffer.

A focused scene test will verify the two friction coefficients, collider ownership, and CCD configuration. Existing settling and grasp/release tests will continue to verify that the low-friction table remains stable and that the higher-friction gripper can still lift and release the cloth. The final interactive run will use the existing `cloth_limx_franka` example so the reduced table drag can be judged visually.

## Non-goals

- Changing cloth self-collision friction.
- Adding velocity or normal damping.
- Introducing per-shape friction to the general contact solver API.
- Changing the grasp trajectory, contact thickness, stiffness, or time step.
