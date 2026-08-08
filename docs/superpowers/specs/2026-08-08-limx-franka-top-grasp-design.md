# LIMX Franka Top-Grasp Design

## Goal

Add a separate Franka scene that grasps a flat square cloth from directly above at its center. The scene must demonstrate contact-driven pinching and lifting rather than the existing side/edge grasp.

## Scene Geometry

The scene reuses the existing `50 x 50` LIMX square cloth resolution, 0.4 m width, 3 mm particle/contact thickness, 0.2 kg mass, and table geometry. The cloth is centered at `(0.0, -0.5)` so every vertex begins above the table; it has no edge overhang.

The Franka base moves to `(-0.3, -1.15, -0.1)` to make the table center reachable with a top-down wrist pose. The gripper orientation is a 180-degree rotation about world X, quaternion `(1, 0, 0, 0)`. Initial IK uses repeated solve batches so the first rendered pose already matches the approach target and does not sweep through the scene.

At the grasp target, the TCP is at `(0.0, -0.5, 0.211)`. Geometry calibration with the selected finger-pad boxes places their lowest vertices at approximately `z = 0.2051 m`, above the table top at `z = 0.2 m` and level with the cloth. The fingers therefore close laterally around the cloth without entering the table.

## Motion Sequence

The open gripper starts at the table center and `z = 0.35 m`. It then:

1. holds above the cloth for 0.5 s;
2. descends vertically to the calibrated grasp height over 0.8 s;
3. closes from `0.04 m` to `0.0029 m` over 0.8 s, gathering the flat center patch into a small ridge;
4. pauses closed at the table for 0.4 s;
5. lifts vertically to `z = 0.42 m` over 1.2 s;
6. holds the lifted cloth for 0.8 s;
7. opens to release it over 0.8 s; and
8. retracts upward over 0.6 s.

All keyframes retain the top-down orientation. Interpolation remains continuous and uses the existing IK and kinematic rigid velocity calculation.

## Contact Model

The scene uses the current LIMX contact operators without artificial attachment:

- cloth self-collision: 3 mm thickness and friction `0.4`;
- table contact: friction `0.05`, normal damping `0.0`, and CCD disabled;
- finger contact: friction `0.4`, normal damping `0.0`, and CCD enabled;
- finger-pad collision geometry: the existing two box proxies and VF/EE force, HVP, and diagonal Hessian paths.

The solver keeps `dt = 0.01 s`, one substep, and the existing penalty-force plus Hessian response. No vertex binding, suction constraint, pre-made cloth crease, or velocity damping is added to force a successful grasp.

## Code Structure

The existing `cloth_limx_franka` scene remains the side-grasp example. Its `Example` constructor gains only private configuration keywords for the Franka base, cloth center, and number of initial IK solve batches, all defaulting to current behavior.

The new `newton/examples/cloth/example_cloth_limx_franka_top_grasp.py` subclasses that example, supplies the centered cloth and closer base, overrides the keyframe sequence, and adds center-patch lift metrics. This keeps collision setup and simulation data flow identical between side and top grasp while avoiding a copied solver integration.

## Validation

A focused CPU test verifies the top-grasp keyframes, constant downward orientation, centered cloth, and calibrated heights. A focused CUDA geometry test verifies the initial IK pose reaches the requested TCP and that both complete finger boxes remain above the table at the grasp pose. A visual rollout is the primary acceptance test: the fingers must close around the center patch, lift visible cloth above the table, hold it, and release it without table penetration.

The existing side-grasp tests remain unchanged and serve as regression coverage for the new constructor defaults.

## Non-goals

- Replacing the side-grasp scene.
- Adding a robot dynamics solver; the Franka remains kinematic input to LIMX.
- Guaranteeing grasp through post-close vertex binding.
- Increasing the cloth to `100 x 100` before the top-grasp interaction is validated.
