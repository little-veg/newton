# LIMX Box Normal-Cone Contact Design

## Goal

Prevent active LIMX cloth from becoming numerically locked to a kinematic box
edge or vertex after the box moves away. Preserve the existing 3 mm contact
thickness, VF/EE representation, friction model, 0.01 s step, and complete
force/Hessian operator.

The motivating Franka rollout must still pinch and lift the cloth. It must
form a graspable fold without driving the finger boxes vertically through the
flat sheet. After the fingers open, the cloth must release instead of
remaining supported by invalid contacts or a topological loop around a finger.

## Root Cause

Box face triangles have outward winding and cloth-vertex/rigid-face contact is
one-sided. Rigid-vertex/cloth-face and cloth-edge/rigid-edge contact currently
use only primitive distance and the step-start separation direction. They do
not know the outward normal cone of the rigid vertex or edge.

Near a lower box edge, a cloth point in the side-face Voronoi region can
therefore also activate the lower edge and its endpoint vertices. Those
redundant contacts have upward components and can balance gravity. The result
looks adhesive even though every individual force is repulsive. Setting
friction to zero does not remove the lock.

The normal-cone correction removes those invalid primitive contacts, but the
original Franka path has a second independent problem: the TCP descends to the
cloth plane while the finger boxes extend another 5.9 mm below it. The boxes
therefore pass vertically through the flat cloth before closing, leaving the
sheet wrapped around a finger. A closed oriented collision model cannot undo
that topology after it has formed.

## Chosen Design

Add box-only rigid-feature ownership metadata to
`ConstraintKinematicMeshContact`:

- each box vertex receives the three outward normals of its incident faces;
- each physical box edge receives the two outward normals of its incident
  faces;
- each coplanar face diagonal receives the single outward face normal;
- mesh and convex-mesh features retain the existing two-sided behavior.

Store the normals in collider-local coordinates and rotate them with the same
body and shape transforms used for collider vertices. A feature with no cone
metadata remains unfiltered for backward compatibility.

Before emitting a rigid-vertex/cloth-face or cloth-edge/rigid-edge contact,
compute the vector from the rigid feature toward the closest cloth point. The
candidate belongs to the rigid feature only when that vector lies in the
feature's outward normal cone. For the orthogonal box normals this requires:

1. a nonnegative coefficient along every stored outward normal, within a
   scale-aware numerical tolerance; and
2. no significant residual outside the span of those normals.

Evaluate ownership for both the current and predicted closest pair. Keep a
predictive candidate when either state has valid ownership, matching the
existing swept-contact policy. Cloth-vertex/box-face contact continues to use
the face winding and one-sided signed gap.

Normal-cone filtering changes only which contact stencils are frozen. Accepted
contacts continue to use the existing normal force, damping, friction,
matrix-free Hessian-vector product, and diagonal Hessian implementations.

## Table-Edge Fold Trajectory

Replace the vertical center penetration with a surface sweep:

1. Keep the fingers open and move the TCP above the front table edge, outside
   the cloth footprint.
2. Descend until the finger-box bottoms remain at least one 3 mm contact layer
   above the tabletop.
3. Translate horizontally from the table edge toward the cloth center. The
   leading box contacts the front cloth edge and pushes it into a fold; no box
   moves vertically through the sheet.
4. Close the fingers at the center to pinch the generated fold.
5. Lift and hold with the existing 0.01 s single-step simulation.
6. Open at the raised pose, then retreat horizontally toward the table edge so
   the fingers leave the fold rather than remaining threaded through it.

The sweep height is derived from the table top, 3 mm contact thickness, and
the selected finger-box lower extent. The test must verify zero cloth/finger
triangle intersections before closing; successful lift is not accepted if the
trajectory first tunnels through the cloth.

## Alternatives Rejected

- **Disable reverse VF and EE for boxes:** avoids the observed lock but misses
  legitimate vertex and edge collisions.
- **Lower friction or collision thickness:** friction zero still reproduces
  the problem, while a smaller activation distance only hides invalid feature
  ownership.
- **Keep the vertical center descent and change only release:** cannot undo the
  cloth/finger loop created before closure.
- **Raise the vertical center grasp:** avoids intersection, but the flat sheet
  never enters the jaws and the measured lift remains zero.

## Validation

Follow test-driven development:

1. Add a box-bottom regression where a cloth triangle lies just inside the
   bottom face. The outward face contact may push it down, but bottom vertices
   viewed from their inward side must not emit reverse VF contacts.
2. Add the corresponding valid exterior case below the bottom face and verify
   that reverse VF remains active.
3. Add invalid and valid box-edge cases to verify that EE candidates are
   rejected outside and retained inside the edge normal cone.
4. Verify the edge-to-center sweep has zero strict cloth/finger triangle
   intersections immediately before closure.
5. Extend the Franka rollout through its open-and-retreat phase and verify that
   the formerly locked raised patch falls away while the earlier closed phase
   still lifts and holds the cloth.
6. Run all kinematic-contact tests, both Franka examples, the registered
   example smoke tests, and `uvx pre-commit run -a`.

The implementation must retain zero contact-buffer overflow, finite state,
one 0.01 s step per frame, and the existing geometric non-penetration bounds.

## Scope and Compatibility

This is an internal behavior correction with no public API change and no new
dependency. It applies normal-cone ownership only to generated box topology;
arbitrary mesh winding is not assumed to define a closed consistently oriented
solid. The user-facing behavior change will be recorded in the Unreleased
`Fixed` section of `CHANGELOG.md`.
