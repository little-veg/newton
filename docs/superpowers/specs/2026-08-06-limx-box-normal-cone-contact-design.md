# LIMX Box Normal-Cone Contact Design

## Goal

Prevent active LIMX cloth from becoming numerically locked to a kinematic box
edge or vertex after the box moves away. Preserve the existing 3 mm contact
thickness, VF/EE representation, friction model, 0.01 s step, and complete
force/Hessian operator.

The motivating Franka rollout must still pinch and lift the cloth. After the
fingers open, the cloth must release instead of remaining supported by
contacts generated from the inward side of a box feature.

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

## Alternatives Rejected

- **Disable reverse VF and EE for boxes:** avoids the observed lock but misses
  legitimate vertex and edge collisions.
- **Lower friction or collision thickness:** friction zero still reproduces
  the problem, while a smaller activation distance only hides invalid feature
  ownership.
- **Change only the Franka release trajectory:** can pull the collider away
  from the symptom but leaves the contact defect in every box scene.

## Validation

Follow test-driven development:

1. Add a box-bottom regression where a cloth triangle lies just inside the
   bottom face. The outward face contact may push it down, but bottom vertices
   viewed from their inward side must not emit reverse VF contacts.
2. Add the corresponding valid exterior case below the bottom face and verify
   that reverse VF remains active.
3. Add invalid and valid box-edge cases to verify that EE candidates are
   rejected outside and retained inside the edge normal cone.
4. Extend the Franka rollout through its existing open phase and verify that
   the formerly locked raised patch falls away while the earlier closed phase
   still lifts and holds the cloth.
5. Run all kinematic-contact tests, both Franka examples, the registered
   example smoke tests, and `uvx pre-commit run -a`.

The implementation must retain zero contact-buffer overflow, finite state,
one 0.01 s step per frame, and the existing geometric non-penetration bounds.

## Scope and Compatibility

This is an internal behavior correction with no public API change and no new
dependency. It applies normal-cone ownership only to generated box topology;
arbitrary mesh winding is not assumed to define a closed consistently oriented
solid. The user-facing behavior change will be recorded in the Unreleased
`Fixed` section of `CHANGELOG.md`.
