# Mobile ALOHA Dual-Arm IK Import Design

## Goal

Add a Newton robot example that imports AgileX Robotics' Split Mobile ALOHA,
renders the complete platform, and controls both Piper arms through independent
interactive end-effector targets. The mobile base, steering, wheels, lift, and
sensors remain structurally fixed. Newton's GPU IK produces joint position
targets, while `SolverMuJoCo` applies force-generating position actuators and
integrates the articulated dynamics.

The runnable command is:

```bash
python -m newton.examples robot_mobile_aloha
```

## Scope

The first milestone includes:

- the complete Split Mobile ALOHA visual model from the upstream
  `split_aloha` branch;
- a fixed chassis with fixed steering, wheels, lift, and sensor mounts;
- two independently targeted six-DoF Piper arms;
- two independently controlled parallel grippers;
- one coordinated IK solve containing both arm TCP objectives;
- MuJoCo joint-drive dynamics under gravity;
- two viewer gizmos, two gripper-opening sliders, and live TCP error values;
- a deterministic headless motion used by `test_final()`;
- cached, revision-pinned upstream asset download with a local path override.

The first milestone does not include:

- mobile-base, steering, wheel, or lift control;
- robot self-collision;
- grasp objects, external contact, cloth, LIMX, or ABD coupling;
- motion planning, collision-aware IK, sensing, perception, or autonomy;
- direct torque control or end-effector force control;
- redistribution of upstream URDF or mesh files inside the Newton repository.

## Upstream Asset and Provenance

Use the repository
`https://github.com/agilexrobotics/mobile_aloha_sim.git` at the immutable
commit
`594da182508f0780a1a81a40494552564babec93`. The combined model is
`split_aloha_mid_360/urdf/split_aloha_mid_360_with_piper.urdf` and references
the sibling ROS packages `split_aloha_mid_360` and `piper_description`.

The upstream tree contains 32 links and 31 top-level URDF joints: 16 revolute,
4 continuous, 5 prismatic, and 6 fixed. Its 47 DAE/STL mesh files occupy about
179 MB. Both ROS package manifests declare `BSD`, but the repository does not
contain a complete root license text. Newton therefore records the source URL
and revision and downloads the files into the user's cache at runtime; it does
not copy the files into the Newton source tree or package artifacts.

The example accepts `--asset-root PATH`. A valid root contains both
`split_aloha_mid_360/` and `piper_description/`. When the override is absent,
the example obtains the repository root through the cached Git downloader. A
failed download reports the pinned URL and revision and tells the user how to
use `--asset-root`; an invalid override reports the first missing package,
URDF, or mesh path.

## Public Download Utility

Expose the existing Git cache helper as
`newton.utils.download_git_folder()`. Extend its accepted folder path so `.`
means the repository root. Root downloads retain the existing immutable-SHA,
content-addressed cache layout, TTL behavior, offline cache reuse, sparse Git
transport, and `force_refresh` semantics.

This public export prevents the example from importing `newton._src` and avoids
creating a second cache implementation. Adding the symbol requires its public
API documentation to be regenerated and an `Added` entry in the
`CHANGELOG.md` `[Unreleased]` section.

## URDF Normalization

Read the combined URDF as XML and create an in-memory normalized XML string
before calling `ModelBuilder.add_urdf()`. Do not modify either the downloaded
cache checkout or a user-provided asset checkout.

Normalization performs two operations:

1. Change these nine joints to `fixed`:
   `fr_steering_joint`, `fr_wheel`, `fl_steering_joint`, `fl_wheel`,
   `rl_steering_joint`, `rl_wheel`, `rr_steering_joint`, `rr_wheel`, and
   `lifting_joint`.
2. Resolve every `package://<package>/<path>` filename attribute to an absolute
   file path below the validated asset root.

Every named lock joint must exist exactly once. Every package URI must use one
of the two validated packages and resolve to an existing file. Missing or
duplicate lock joints, unsupported package names, and missing referenced files
raise descriptive errors before model construction.

Import the normalized XML with `floating=False`,
`collapse_fixed_joints=True`, `enable_self_collisions=False`, and
`parse_visuals_as_colliders=False`. Fixed-joint collapsing preserves the full
visual geometry while removing the base-mechanism degrees of freedom. The
remaining actuated joints are exactly the twelve arm revolute joints and four
finger prismatic joints.

## Initial State and Joint Mapping

Enable coordinate-layout targets with
`newton.use_coord_layout_targets = True`. Resolve bodies and joints by their
labels rather than relying on importer order. Require the following active
joint labels:

- `left/joint1` through `left/joint6`;
- `left/joint7` and `left/joint8`;
- `right/joint1` through `right/joint6`;
- `right/joint7` and `right/joint8`.

Initialize both arm chains at the compact, limit-safe pose
`[0.0, 1.2, -1.2, 0.0, 0.0, 0.0]` rad. Initialize each gripper to a total
opening of `0.07 m`, represented by finger coordinates `+0.035 m` and
`-0.035 m`. Copy this state into `joint_target_q` before creating the first
simulation state so the dynamics do not receive a startup step input.

The left and right TCPs are points on `left/link6` and `right/link6` with the
local offset `(0.0, 0.0, 0.13503) m`, matching the two finger-joint origins in
the combined URDF. Each TCP orientation is the corresponding `link6` frame
orientation. Initial gizmo transforms come from FK at this initial state.

## Dual-Arm IK

Use one `IKSolver` with `n_problems=1`, the analytic Jacobian, LM optimizer,
sampling disabled, and 24 iterations per rendered frame. Its ordered
objectives are:

1. left TCP position;
2. left TCP rotation;
3. right TCP position;
4. right TCP rotation;
5. the model-wide joint-limit objective.

A model-wide `joint_dof_mask` is `True` only for the twelve revolute arm DoFs.
The fingers are excluded because their targets come from UI sliders, and the
normalized base mechanisms no longer have DoFs. Warm-start every solve from
the preceding frame's IK solution.

After every IK solve:

1. reject the result if any optimized coordinate is non-finite and retain the
   last valid command;
2. clamp the twelve desired arm coordinates to their URDF limits;
3. rate-limit each command relative to the previous commanded target by
   `joint_velocity_limit * frame_dt`;
4. copy only those twelve coordinates into `control.joint_target_q`;
5. leave `control.joint_target_qd` at zero.

An unreachable gizmo target remains visible. IK returns its best bounded
configuration, while the UI reports the remaining position error in meters and
rotation error in radians for each arm. The example does not silently move,
clamp, or replace the user's Cartesian target.

## Gripper Control

Provide independent `Left opening` and `Right opening` sliders with the range
`[0.0, 0.1] m`. For an opening `w`, command the corresponding finger pair as
`(+0.5*w, -0.5*w)`, then clamp each coordinate to its imported URDF limit.
The sliders start at `0.07 m`. Gripper coordinates are never part of the IK
solve.

## MuJoCo Dynamics

Use `SolverMuJoCo` with Newton joint-target control, the `newton` solver,
`implicitfast` integration, and ten simulation substeps per 60 Hz rendered
frame. Disable contacts for this control-only milestone. Gravity remains
`(0.0, 0.0, -9.81) m/s^2`, so successful tracking demonstrates that the
actuators support the arms rather than merely replaying kinematics.

Seed each arm's position gains from the Piper upstream MuJoCo model:

```text
joint:       1      2      3     4    5    6
target_ke: 10000   2000   2000   500  200  200
damping:     500      5     20     5    5    5
```

The damping values are passive `joint_damping`, matching the source model;
`joint_target_kd` remains zero so damping is not applied twice. Each finger
uses `target_ke=10000` and passive damping `100`. Preserve the combined URDF's
effort and coordinate limits. `JointTargetMode.POSITION` is used for all
sixteen active DoFs.

The high source gains are paired with the joint-target rate limiter. They may
only be changed if the exact imported model demonstrates instability or failure
to track during the specified tests; any change must be documented against the
upstream values rather than introduced as an unexplained tuning constant.

## Per-Frame Data Flow

For every rendered frame:

1. read the two gizmo transforms and two gripper-opening sliders;
2. update the four Cartesian IK targets;
3. run the joint-masked dual-arm IK solve;
4. validate, clamp, and rate-limit the twelve arm targets;
5. map both gripper openings into four finger targets;
6. advance `SolverMuJoCo` for ten substeps with the same control targets;
7. evaluate both TCP transforms and their target errors from the resulting
   dynamic state;
8. render the complete robot, gizmos, and UI error values.

The viewer Reset action reconstructs only simulation and controller state. It
reuses the cached asset checkout and must not re-download the 179 MB upstream
repository when the pinned revision is already available.

## Components and Files

Implementation is confined to these responsibilities:

- `newton/_src/utils/download_assets.py`: accept and cache repository-root Git
  downloads;
- `newton/_src/utils/__init__.py` and `newton/utils.py`: export
  `download_git_folder` publicly;
- `newton/examples/robot/example_robot_mobile_aloha.py`: asset validation,
  in-memory URDF normalization, model construction, joint mapping, IK,
  controls, simulation, rendering, and example-level assertions;
- `newton/tests/test_download_assets.py`: repository-root cache behavior;
- `newton/tests/test_example_robot_mobile_aloha.py`: offline normalization,
  mapping, clamping, and gripper-target unit tests;
- `newton/tests/test_examples.py`: CUDA example smoke registration;
- `README.md` and
  `docs/images/examples/example_robot_mobile_aloha.jpg`: command registration
  and 320-by-320 screenshot;
- generated public API documentation and the `[Unreleased]` changelog entry.

No new required or optional dependency is added. The implementation uses
Warp, NumPy, Python's XML standard library, and Newton's existing GitPython
dependency.

## Tests and Acceptance

Use `unittest` and test-first implementation.

### Offline Unit Tests

1. Create a local temporary Git repository with two nested package folders,
   record its `repo.head.commit.hexsha` as `local_commit`, and verify
   `download_git_folder(..., folder_path=".", ref=local_commit)` returns the
   cached repository root, preserves nested files, and reuses the exact cache
   while the remote is unavailable.
2. Normalize a synthetic URDF containing all nine base-mechanism joints, one
   left arm joint, one right arm joint, and package mesh references. Verify the
   nine selected types become `fixed`, arm types remain unchanged, every URI
   becomes an absolute existing path, and the input file is unchanged.
3. Verify missing and duplicate lock joints, unsupported packages, and missing
   mesh files each raise a descriptive exception.
4. Verify joint-label lookup rejects missing or duplicate required labels.
5. Verify arm target clamping and rate limiting respect per-joint coordinate
   and velocity limits.
6. Verify a `0.1 m` gripper opening maps to `(+0.05, -0.05) m` and smaller
   imported limits clamp both fingers symmetrically.

### Full CUDA Example Test

The first full run downloads the pinned upstream revision; later runs use its
cache. In `--test` mode, initialize both targets from FK, then command a
reachable `0.02 m` upward TCP translation while preserving each initial TCP
rotation. Hold the target for 180 rendered frames.

`test_final()` requires:

- finite joint coordinates, joint velocities, body transforms, and command
  targets;
- the imported active joint set to contain exactly twelve revolute arm DoFs
  and four prismatic finger DoFs;
- every arm coordinate to remain inside its imported limit with `1e-5`
  tolerance;
- each final TCP position error below `0.02 m` and rotation error below
  `0.10 rad`;
- the fixed root transform to match its initial transform within `1e-6`;
- each finger pair to remain opposite in sign with magnitude mismatch below
  `1e-5 m`;
- every per-frame arm target increment to stay below its velocity-limit bound
  with `1e-5` tolerance.

The test must fail if IK is bypassed by assigning dynamic joint state directly.
The example updates only `Control` targets; `State.joint_q` and
`State.joint_qd` are outputs of `SolverMuJoCo`.

### Visual Acceptance

Run the GL viewer on CUDA and verify:

- chassis, wheels, lift, cameras, both arms, and both grippers have correct
  relative scale and placement;
- neither arm jumps when the viewer opens or resets;
- dragging either gizmo moves only its intended arm while both targets remain
  jointly satisfied;
- both gripper sliders open and close the intended finger pair;
- displayed TCP errors decrease after a reachable target move;
- the base, wheels, steering, and lift remain stationary.

Capture a 320-by-320 screenshot after these checks and register the example in
the Robot Examples section of `README.md`.

## Future Work

Later designs may enable robot collision, add grasp objects, connect the arm
links to LIMX cloth or ABD contact, or restore the mobile-base and lift DoFs.
Those extensions must preserve the current boundary: IK generates targets,
PD actuators generate generalized forces, and the dynamics solver owns the
actual joint state.

## References

- AgileX Robotics Split Mobile ALOHA source and combined URDF at commit
  `594da182508f0780a1a81a40494552564babec93`.
- Upstream `piper_description/mujoco_model/piper_description.xml` for arm
  position gains and passive damping.
- Newton `example_robot_panda_hydro.py` for IK-to-PD-to-MuJoCo control flow.
- Newton `example_ik_franka.py` for interactive gizmo target handling.
- Newton `IKSolver.joint_dof_mask` for excluding non-arm DoFs from LM updates.
