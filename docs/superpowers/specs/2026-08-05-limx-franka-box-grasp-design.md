# LIMX Franka Box-Grasp Design

## Goal

Turn the existing fixed-cloth Franka motion scaffold into a physical one-way
grasp demonstration. A kinematically driven Franka must close its box collision
proxies around active LIMX cloth, lift the cloth by frictional VF/EE contact,
and hold it without an attachment constraint. The simulation step is fixed at
0.01 s with one solver step per rendered 100 Hz frame.

## Scope

This milestone includes:

- active square cloth with LIMX membrane, bending, and self-collision terms;
- kinematic Franka motion using the existing approach, close, lift, and hold
  sequence;
- box collision proxies for both fingers and the existing rigid collision
  shapes for the wrist, hand, and table;
- moving-collider VF and EE normal contact, damping, friction, HVP, and diagonal
  Hessian contributions;
- step-level predictive candidate detection with one-sided contact orientation;
- geometric intersection checks against the actual collision boxes;
- an interactive visualization of the grasp attempt.

This milestone does not add visual-mesh collision, artificial grasp
attachments, rigid-body reaction wrenches, two-way rigid-soft coupling, or
robot actuator dynamics. The rendered finger mesh is larger than its box
proxy, so collision acceptance is measured against the boxes. The example
will expose or render the collision surface clearly enough to distinguish a
box contact failure from visual-mesh overlap.

## Chosen Architecture

Use one Newton model for the Franka, table, and cloth, but treat rigid bodies as
prescribed kinematic data during the LIMX solve. The existing IK controller
produces arm and finger joint coordinates. Forward kinematics writes the
corresponding rigid transforms and velocities before each cloth step.
`ConstraintKinematicMeshContact` then updates its triangulated box surface from
that rigid state. LIMX advances only the cloth and carries the rigid and joint
state through unchanged.

This is preferred over full two-way coupling because the immediate question is
whether box VF/EE contact and friction can capture and carry cloth. It is also
preferred over an attachment constraint because an attachment would make the
lift succeed without validating collision or friction.

## Predictive One-Sided Contact

The current kinematic contact detector rebuilds unsigned contacts only around
the current Newton iterate. At a 0.01 s step, an iterate can cross a thin box
surface before a contact is detected. The unsigned normal then flips to the
new side, allowing an intersecting state to settle. Increasing collision
thickness or adding substeps is not the chosen remedy.

At `begin_step()`, cache cloth positions, cloth velocities, collider positions,
collider velocities, and the 0.01 s step. Build conservative swept AABBs from
the step-start and linearly predicted endpoints for both cloth and collider
triangles and edges. Query those swept bounds during every nonlinear iteration.
This keeps VF and EE candidate pairs available after an unconstrained Newton
increment crosses a surface.

For each candidate, derive its side orientation from the step-start
configuration and preserve that orientation for the entire step. At the
current Newton iterate, evaluate a signed gap along the preserved direction:

```text
depth = thickness - signed_gap
```

Normal force and its rank-one Hessian are active only when `depth > 0`.
Crossing the surface therefore increases depth instead of flipping the normal.
Approaching-only damping can activate for a swept candidate predicted to enter
the contact zone; its force, HVP, and diagonal continue to use the same frozen
relative-motion model. Friction activates only under positive normal load.

Apply the same rule to cloth-vertex/rigid-face,
rigid-vertex/cloth-face, and cloth-edge/rigid-edge candidates. Degenerate
features or a step-start configuration without a stable separation direction
remain excluded. The existing IPC near-parallel EE mollifier continues to
scale EE normal, damping, friction, and second-order terms consistently.

## Grasp Sequence and Data Flow

The example reuses the existing keyframed path, adjusted for 100 Hz:

1. Hold the open gripper above the cloth.
2. Descend around the flat cloth while the table supports it.
3. Close the two finger box proxies around the cloth.
4. Hold closed briefly so frictional contact settles.
5. Lift the hand vertically by at least 0.10 m while remaining closed.
6. Hold the raised pose for at least 0.5 s.

For every frame:

1. Interpolate the current TCP and finger targets.
2. Solve IK and compute the target joint velocity.
3. Evaluate FK to update body transforms and spatial velocities.
4. Update the kinematic collision surface.
5. Advance active cloth by one LIMX step of 0.01 s.
6. Record contact, collision-intersection, and cloth-lift metrics.
7. Render the Franka, table, cloth, and inspectable collision proxies.

The table and robot share the same kinematic contact operator. Table vertices
have world body index `-1` and zero velocity; Franka boxes follow FK.

## Physical Parameters

Retain the established cloth scale and baseline contact parameters:

- 0.4 m square, 21-by-21 vertices, and 0.2 kg total cloth mass;
- 3 mm cloth and rigid contact thickness;
- 20 kN/m normal stiffness and 0.5 N·s/m normal damping;
- friction coefficient 0.4 with 0.01 m/s regularization;
- two nonlinear iterations and 50 PCG iterations;
- velocity damping 0.998;
- 100 Hz, `dt = 0.01 s`, and no substeps.

Only one physical parameter is changed at a time if the baseline pinch fails.
Friction may be tuned after geometric non-crossing is established; stiffness,
thickness, or hidden substeps must not be used to mask a predictive-contact
failure.

## Validation

Follow test-driven development. First add a synthetic high-speed contact test
that fails with the current discrete unsigned detector: a cloth primitive must
not cross a static triangulated box face during one 0.01 s step. The test must
exercise the public constraint through `SolverLIMX`, rather than asserting
private buffer contents alone.

Focused contact tests then verify:

- swept VF candidates remain oriented to the step-start side;
- swept EE candidates retain a stable direction across an attempted crossing;
- inactive predictive candidates do not add normal force or normal Hessian;
- active one-sided candidates add consistent force, HVP, and diagonal blocks;
- moving collider velocity contributes to predictive damping;
- graph capture and fixed-capacity overflow accounting still work.

The grasp rollout succeeds only if all of the following hold:

- state remains finite and no contact buffer overflows;
- cloth and box collision triangles have no intersections after closure;
- the cloth centroid rises at least 0.10 m above its pre-lift height;
- the cloth remains captured for at least 0.5 s at the raised pose;
- the simulation uses exactly one 0.01 s step per frame.

The final acceptance check is an interactive GL run. If the baseline contact
does not lift the cloth, preserve the geometric and contact diagnostics and
tune only the demonstrated limiting parameter.
