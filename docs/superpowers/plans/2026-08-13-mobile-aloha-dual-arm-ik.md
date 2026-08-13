# Mobile ALOHA Dual-Arm IK Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable Newton example that downloads and imports the complete Split Mobile ALOHA model, fixes the mobile mechanisms, and dynamically tracks two independent TCP gizmos with joint-masked IK and MuJoCo PD drives.

**Architecture:** Reuse Newton's content-addressed Git cache through a new public export, normalize the upstream URDF entirely in memory, and keep pure asset/control helpers independently testable. The example solves both arms in one LM IK problem, rate-limits the resulting arm targets, maps independent gripper openings to paired finger targets, and advances only `SolverMuJoCo` state.

**Tech Stack:** Python 3.12, Warp, NumPy, Newton public APIs, MuJoCo Warp, GitPython, `xml.etree.ElementTree`, `unittest`.

## Global Constraints

- Work only in the canonical `/home/limx/github/newton` checkout on branch `dev`; do not create a worktree.
- Preserve the existing modified `docs/images/examples/example_cloth_limx_three_tshirts_box.jpg` and untracked `solver_convergence.png`.
- Do not import `newton._src` from examples or docs.
- Add no required or optional dependency.
- Use `newton.use_coord_layout_targets = True` and index targets through `joint_q_start` / `joint_qd_start`.
- Use `unittest`, give every test a triple-double-quoted imperative docstring, and do not call a Warp synchronize function before `.numpy()`.
- Keep the base, four steering joints, four wheel joints, and lift structurally fixed; only twelve arm revolute joints and four finger prismatic joints remain active.
- Use the upstream repository at commit `594da182508f0780a1a81a40494552564babec93`; do not redistribute its URDF or mesh files in Newton.
- Use joint-target PD dynamics through `SolverMuJoCo`; never assign solved IK coordinates to `State.joint_q` or `State.joint_qd`.
- Keep contacts and robot self-collision disabled in this milestone.
- Register the example in `README.md` with a 320-by-320 screenshot and add a random-position `[Unreleased] / Added` changelog entry.

---

### Task 1: Public repository-root Git cache

**Files:**
- Modify: `newton/tests/test_download_assets.py`
- Modify: `newton/_src/utils/__init__.py`
- Modify: `newton/utils.py`
- Modify: `docs/api/newton_utils.rst` through `docs/generate_api.py`

**Interfaces:**
- Consumes: existing `download_git_folder(git_url: str, folder_path: str, cache_dir: str | None = None, ref: str = "main", force_refresh: bool = False) -> Path`.
- Produces: public `newton.utils.download_git_folder` with verified `folder_path="."` repository-root behavior.

- [ ] **Step 1: Write the failing public-API root-download test**

Add this method to `TestDownloadAssets`:

```python
def test_public_root_download_by_commit(self):
    """Download and reuse a complete repository root through the public API."""
    import newton.utils

    nested = Path(self.work_dir, "piper_description", "meshes")
    nested.mkdir(parents=True)
    (nested / "link.stl").write_text("mesh\n", encoding="utf-8")
    self.work.index.add([str(nested / "link.stl")])
    commit = self.work.index.commit("add nested package").hexsha
    self.work.git.push("origin", "main")

    root = newton.utils.download_git_folder(
        self.remote_dir, ".", cache_dir=self.cache_dir, ref=commit
    )
    self.assertEqual((root / "piper_description/meshes/link.stl").read_text(), "mesh\n")

    with mock.patch(
        "newton._src.utils.download_assets._get_latest_commit_via_git",
        return_value=None,
    ):
        cached = newton.utils.download_git_folder(
            self.remote_dir, ".", cache_dir=self.cache_dir, ref=commit
        )
    self.assertEqual(cached.resolve(), root.resolve())
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run --extra dev -m newton.tests -p test_download_assets.py -k test_public_root_download_by_commit
```

Expected: FAIL because `newton.utils` has no `download_git_folder` attribute.

- [ ] **Step 3: Export the existing helper publicly**

Import and list `download_git_folder` in both public utility modules:

```python
from .download_assets import clear_git_cache, download_asset, download_git_folder
```

and:

```python
from ._src.utils.download_assets import download_asset, download_git_folder
```

Add the name beside `download_asset` in each `__all__`. Do not duplicate or wrap the existing cache implementation.

- [ ] **Step 4: Run the test and verify GREEN**

Run the Step 2 command. Expected: PASS, including the nested file and cached-path assertions.

- [ ] **Step 5: Regenerate public API documentation and rerun focused tests**

Run:

```bash
uv run docs/generate_api.py
uv run --extra dev -m newton.tests -p test_download_assets.py
```

Expected: the download-assets module passes and `docs/api/newton_utils.rst` lists `download_git_folder`.

- [ ] **Step 6: Commit the public utility**

```bash
git add newton/tests/test_download_assets.py newton/_src/utils/__init__.py newton/utils.py docs/api/newton_utils.rst
git commit -m "Expose cached Git folder downloads"
```

---

### Task 2: Mobile ALOHA asset and command helpers

**Files:**
- Create: `newton/examples/robot/example_robot_mobile_aloha.py`
- Create: `newton/tests/test_example_robot_mobile_aloha.py`

**Interfaces:**
- Produces: `resolve_mobile_aloha_asset_root(asset_root: str | Path | None) -> Path`.
- Produces: `normalize_mobile_aloha_urdf(asset_root: str | Path) -> str`.
- Produces: `find_unique_label(labels: Sequence[str], required: str) -> int`.
- Produces: `clamp_and_rate_limit_targets(solution, previous, lower, upper, velocity, frame_dt) -> np.ndarray`.
- Produces: `gripper_joint_targets(opening, lower, upper) -> np.ndarray`.

- [ ] **Step 1: Write failing helper tests with a synthetic two-package tree**

Create `newton/tests/test_example_robot_mobile_aloha.py` with a loader for
`newton.examples.robot.example_robot_mobile_aloha` and a `TemporaryDirectory`
containing `split_aloha_mid_360/urdf`, `split_aloha_mid_360/meshes`, and
`piper_description/meshes`. Build a compact URDF containing every name in
`LOCKED_JOINT_NAMES`, one `left/joint1`, one `right/joint1`, and package mesh
references. Add these tests:

```python
def test_normalize_mobile_aloha_urdf(self):
    """Fix only base mechanisms and resolve package meshes absolutely."""
    xml = self.module.normalize_mobile_aloha_urdf(self.asset_root)
    root = ET.fromstring(xml)
    types = {joint.get("name"): joint.get("type") for joint in root.findall("joint")}
    self.assertTrue(all(types[name] == "fixed" for name in self.module.LOCKED_JOINT_NAMES))
    self.assertEqual(types["left/joint1"], "revolute")
    self.assertEqual(types["right/joint1"], "revolute")
    for mesh in root.iter("mesh"):
        self.assertTrue(Path(mesh.get("filename")).is_absolute())
        self.assertTrue(Path(mesh.get("filename")).is_file())

def test_rejects_incomplete_mobile_aloha_assets(self):
    """Reject missing lock joints, unsupported packages, and missing meshes."""
    # Mutate a fresh synthetic tree separately for each subTest and assert
    # ValueError or FileNotFoundError includes the missing name or path.

def test_find_unique_label(self):
    """Resolve one exact label and reject missing or duplicate labels."""
    self.assertEqual(self.module.find_unique_label(["a", "left/joint1"], "left/joint1"), 1)
    with self.assertRaisesRegex(ValueError, "missing"):
        self.module.find_unique_label(["a"], "left/joint1")
    with self.assertRaisesRegex(ValueError, "duplicate"):
        self.module.find_unique_label(["left/joint1", "left/joint1"], "left/joint1")

def test_clamp_and_rate_limit_targets(self):
    """Clamp arm commands to coordinate and per-frame velocity limits."""
    result = self.module.clamp_and_rate_limit_targets(
        np.array([2.0, -2.0]), np.array([0.0, 0.0]),
        np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
        np.array([3.0, 6.0]), 0.1,
    )
    np.testing.assert_allclose(result, [0.3, -0.6])
    stale = self.module.clamp_and_rate_limit_targets(
        np.array([np.nan, 0.0]), np.array([0.2, -0.2]),
        np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
        np.array([3.0, 6.0]), 0.1,
    )
    np.testing.assert_array_equal(stale, [0.2, -0.2])

def test_gripper_joint_targets(self):
    """Map total opening to opposite finger coordinates within limits."""
    result = self.module.gripper_joint_targets(
        0.1, np.array([0.0, -0.04]), np.array([0.04, 0.0])
    )
    np.testing.assert_allclose(result, [0.04, -0.04])
```

- [ ] **Step 2: Run helper tests and verify RED**

```bash
uv run --extra dev -m newton.tests -p test_example_robot_mobile_aloha.py
```

Expected: FAIL because the example module and helper interfaces do not exist.

- [ ] **Step 3: Implement the constants and pure helpers**

In the example module define:

```python
MOBILE_ALOHA_URL = "https://github.com/agilexrobotics/mobile_aloha_sim.git"
MOBILE_ALOHA_REF = "594da182508f0780a1a81a40494552564babec93"
MOBILE_ALOHA_URDF = Path("split_aloha_mid_360/urdf/split_aloha_mid_360_with_piper.urdf")
LOCKED_JOINT_NAMES = (
    "fr_steering_joint", "fr_wheel", "fl_steering_joint", "fl_wheel",
    "rl_steering_joint", "rl_wheel", "rr_steering_joint", "rr_wheel",
    "lifting_joint",
)
```

`resolve_mobile_aloha_asset_root()` validates both package directories and the
combined URDF, or calls public `newton.utils.download_git_folder()` with
`folder_path="."` and the fixed ref. Wrap download failures with an actionable
message containing `--asset-root`.

`normalize_mobile_aloha_urdf()` parses a copy of the source XML, verifies each
lock joint occurs exactly once, sets only those types to `fixed`, rewrites every
`package://` mesh filename to an existing absolute path under one of the two
packages, neutralizes an explicit COLLADA reciprocal-unit compensation scale,
and returns `ET.tostring(..., encoding="unicode")`.

Implement label lookup and NumPy target helpers exactly as exercised by the
tests. Validate equal array shapes, positive `frame_dt`, and nonnegative finite
velocity limits.

- [ ] **Step 4: Run helper tests and verify GREEN**

Run the Step 2 command. Expected: all helper tests PASS without network access.

- [ ] **Step 5: Commit the asset layer**

```bash
git add newton/examples/robot/example_robot_mobile_aloha.py newton/tests/test_example_robot_mobile_aloha.py
git commit -m "Prepare Mobile ALOHA assets"
```

---

### Task 3: Dynamic dual-arm IK example

**Files:**
- Modify: `newton/examples/robot/example_robot_mobile_aloha.py`
- Modify: `newton/tests/test_example_robot_mobile_aloha.py`
- Modify: `newton/tests/test_examples.py`

**Interfaces:**
- Consumes: Task 2 asset/helper functions.
- Produces: `Example(viewer, args)` with `step()`, `render()`, `gui()`, `test_post_step()`, and `test_final()`.
- Produces: CLI option `--asset-root PATH`.

- [ ] **Step 1: Write the failing CUDA integration test**

Add a CUDA-only `TestMobileAlohaExample.test_tracks_both_tcp_targets` that:

```python
def test_tracks_both_tcp_targets(self):
    """Track reachable dual-TCP targets through dynamic joint drives."""
    device = wp.get_cuda_devices()[0]
    asset_root = self.module.resolve_mobile_aloha_asset_root(None)
    args = types.SimpleNamespace(asset_root=str(asset_root), test=True)
    with wp.ScopedDevice(device):
        example = self.module.Example(ViewerNull(num_frames=180), args)
        for _ in range(180):
            example.step()
            example.test_post_step()
        example.test_final()
```

Skip only when CUDA or MuJoCo Warp is unavailable. Also register
`robot.example_robot_mobile_aloha` in `test_examples.py` for CUDA with
`num-frames=180`, `use_viewer=True`, and a 900-second timeout.

- [ ] **Step 2: Run the integration test and verify RED**

```bash
uv run --extra dev -m newton.tests -p test_example_robot_mobile_aloha.py -k test_tracks_both_tcp_targets
```

Expected: FAIL because `Example` is not implemented.

- [ ] **Step 3: Build and configure the normalized model**

Implement `Example.__init__` with:

- 60 Hz frames and ten MuJoCo substeps;
- `newton.use_coord_layout_targets = True`;
- fixed-root normalized URDF import followed by selective fixed-joint collapse
  that retains the generated world-to-base fixed joint, with no self-collision
  and authored collision meshes;
- exact label lookup for twelve arm joints, four fingers, both `link6` bodies,
  and the fixed root;
- initial arm coordinates `[0.0, 1.2, -1.2, 0.0, 0.0, 0.0]` on both sides;
- initial finger coordinates `(+0.035, -0.035)` on both sides;
- per-arm `target_ke=[10000, 2000, 2000, 500, 200, 200]` and passive
  `joint_damping=[500, 5, 20, 5, 5, 5]`;
- per-finger `target_ke=10000`, passive `joint_damping=100`, and
  `JointTargetMode.POSITION` for all active DoFs;
- `SolverMuJoCo(..., disable_contacts=True, solver="newton", integrator="implicitfast")`.

Initialize `state_0`, `state_1`, `control`, and FK only after builder targets
match the initial state.

- [ ] **Step 4: Implement joint-masked dual-arm IK**

Create one LM `IKSolver` with analytic Jacobians, no sampling, and a
`joint_dof_mask` that is true only at the twelve arm DoF indices. Add left and
right position/rotation objectives followed by one joint-limit objective. Use
`link_offset=(0, 0, 0.13503)` for both link6 bodies. Keep a separate
`joint_q_ik` buffer initialized from the model coordinates and solve 24
iterations per frame.

In test mode, move each initial target upward by `0.02 m` without changing its
rotation. In interactive mode, keep the two mutable gizmo transforms as the
targets.

- [ ] **Step 5: Implement command filtering, dynamics, UI, and rendering**

For every frame:

1. copy both gizmo transforms into the four IK objectives;
2. solve IK into the warm-started IK buffer;
3. extract the twelve arm coordinates, call
   `clamp_and_rate_limit_targets()`, and update only their entries in
   `control.joint_target_q`;
4. map both opening sliders through `gripper_joint_targets()`;
5. clear forces and advance `SolverMuJoCo` for ten substeps;
6. compute both actual TCP transforms and position/rotation errors;
7. track maximum target increments and finite-state invariants for tests.

`render()` logs the dynamic state and two gizmos without `snap_to`, preserving
unreachable targets. `gui()` exposes two `[0, 0.1] m` sliders and numeric TCP
position/rotation errors. Set a camera showing the complete robot.

- [ ] **Step 6: Implement example assertions**

`test_post_step()` checks finite state and accumulates maximum commanded arm
increments. `test_final()` asserts exactly twelve active arm DoFs and four
finger DoFs, joint limits within `1e-5`, both position errors below `0.02 m`,
both rotation errors below `0.10 rad`, fixed-root transform drift below
`1e-6`, opposite finger coordinates within `1e-5 m`, and target increments
within velocity bounds plus `1e-5`.

- [ ] **Step 7: Run integration and registered example tests**

```bash
uv run --extra dev -m newton.tests -p test_example_robot_mobile_aloha.py
uv run --extra dev -m newton.tests -p test_examples.py -k example_robot_mobile_aloha
```

Expected: both commands PASS on CUDA; the first asset download is cached and
the second command does not download it again.

- [ ] **Step 8: Commit the dynamic example**

```bash
git add newton/examples/robot/example_robot_mobile_aloha.py newton/tests/test_example_robot_mobile_aloha.py newton/tests/test_examples.py
git commit -m "Add Mobile ALOHA dual-arm IK"
```

---

### Task 4: User-facing registration and visual validation

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Create: `docs/images/examples/example_robot_mobile_aloha.jpg`

**Interfaces:**
- Consumes: runnable `python -m newton.examples robot_mobile_aloha` example.
- Produces: discoverable Robot Examples entry and source-attributed user-facing behavior.

- [ ] **Step 1: Add the README entry before taking the screenshot**

Insert a Robot Examples tile using the command:

```text
python -m newton.examples robot_mobile_aloha
```

and image path
`docs/images/examples/example_robot_mobile_aloha.jpg`. Mention in the example
source header that assets are downloaded from the pinned AgileX repository and
are not distributed by Newton.

- [ ] **Step 2: Add changelog entry at a random Added position**

Use imperative present tense:

```markdown
- Add a dual-arm IK Mobile ALOHA example with cached upstream URDF assets and MuJoCo joint-drive dynamics.
```

- [ ] **Step 3: Launch and inspect the interactive CUDA viewer**

```bash
uv run --extra examples -m newton.examples robot_mobile_aloha --device cuda:0 --num-frames 600
```

Verify complete scale and placement, no startup jump, independent gizmos,
independent gripper sliders, decreasing reachable-target errors, and stationary
base mechanisms.

- [ ] **Step 4: Capture and validate the screenshot**

Capture the rendered robot, crop/resize to exactly 320 by 320 pixels, save it
at the README path, and inspect it visually before staging.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md CHANGELOG.md docs/images/examples/example_robot_mobile_aloha.jpg
git commit -m "Document Mobile ALOHA example"
```

---

### Task 5: Final verification and review

**Files:**
- Verify all files changed by Tasks 1–4.
- Include: `docs/superpowers/plans/2026-08-13-mobile-aloha-dual-arm-ik.md`.

**Interfaces:**
- Consumes: complete feature and focused tests.
- Produces: a clean, reviewable `dev` branch with preserved unrelated user files.

- [ ] **Step 1: Run focused regression tests**

```bash
uv run --extra dev -m newton.tests -p test_download_assets.py
uv run --extra dev -m newton.tests -p test_example_robot_mobile_aloha.py
uv run --extra dev -m newton.tests -p test_examples.py -k example_robot_mobile_aloha
```

- [ ] **Step 2: Run required lint and formatting**

```bash
uvx pre-commit run -a
```

If the repository baseline is reformatted outside this feature, restore only
those unrelated hook edits and run the hooks on the changed feature files as a
second focused check.

- [ ] **Step 3: Review the final diff and repository state**

```bash
git diff --check
git status --short --branch
git log -6 --oneline --decorate
```

Confirm the pre-existing T-shirt screenshot modification and
`solver_convergence.png` remain unstaged and unchanged.

- [ ] **Step 4: Add the ignored implementation plan and commit any final fixes**

```bash
git add -f docs/superpowers/plans/2026-08-13-mobile-aloha-dual-arm-ik.md
git add newton/_src/utils/__init__.py newton/utils.py \
  newton/tests/test_download_assets.py \
  newton/examples/robot/example_robot_mobile_aloha.py \
  newton/tests/test_example_robot_mobile_aloha.py newton/tests/test_examples.py \
  docs/api/newton_utils.rst README.md CHANGELOG.md \
  docs/images/examples/example_robot_mobile_aloha.jpg
git commit -m "Finalize Mobile ALOHA import"
```

Skip the final-fix commit when the working tree contains no uncommitted feature
files; amend no earlier user commit. Do not push until the user explicitly asks.

---

### Task 6: Capture IK and MuJoCo execution on CUDA

**Files:**
- Modify: `newton/tests/test_example_robot_mobile_aloha.py`
- Modify: `newton/examples/robot/example_robot_mobile_aloha.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the existing persistent `joint_q_ik`, IK objectives, MuJoCo solver,
  `state_0`, `state_1`, and `control` arrays.
- Produces: `Example.graph_ik`, `Example.graph_sim`, `Example.capture()`,
  `Example.simulate()`, and a persistent `Example.control_target_q` host array.

- [ ] **Step 1: Require captured execution in the CUDA integration test**

Immediately after constructing `Example` in `test_tracks_both_tcp_targets()`,
add:

```python
self.assertEqual(example.ik_iterations, 24)
self.assertEqual(example.sim_substeps, 10)
self.assertIsNotNone(example.graph_ik)
self.assertIsNotNone(example.graph_sim)
np.testing.assert_allclose(example.control_target_q, example.control.joint_target_q.numpy())
```

After the existing 180-frame loop, repeat the `control_target_q` equality
assertion before `example.test_final()`. This verifies that the graph path keeps
the CPU command cache and GPU control buffer consistent without changing the
validated numerical schedule.

- [ ] **Step 2: Run the integration test and verify RED**

```bash
uv run --extra dev -m newton.tests \
  -p test_example_robot_mobile_aloha.py \
  -k test_tracks_both_tcp_targets
```

Expected: FAIL because the current example has no `graph_ik`, `graph_sim`, or
`control_target_q` attributes.

- [ ] **Step 3: Cache immutable command data during initialization**

In `Example.__init__`, replace repeated per-frame limit downloads with one
initial download and preserve an authoritative host target array:

```python
joint_limit_lower = self.model.joint_limit_lower.numpy()
joint_limit_upper = self.model.joint_limit_upper.numpy()
self.arm_lower_limits = joint_limit_lower[self.arm_dof_indices]
self.arm_upper_limits = joint_limit_upper[self.arm_dof_indices]
self.arm_velocity_limits = self.model.joint_velocity_limit.numpy()[self.arm_dof_indices]
self.finger_lower_limits = joint_limit_lower[self.finger_dof_indices]
self.finger_upper_limits = joint_limit_upper[self.finger_dof_indices]
self.control_target_q = self.control.joint_target_q.numpy()
```

In `_update_commands()`, mutate `self.control_target_q` directly, use the
cached finger-limit arrays, and upload the completed buffer once:

```python
target_q = self.control_target_q
for side in range(2):
    finger_slice = slice(2 * side, 2 * side + 2)
    coord_indices = self.finger_coord_indices[finger_slice]
    target_q[coord_indices] = gripper_joint_targets(
        self.gripper_openings[side],
        self.finger_lower_limits[finger_slice],
        self.finger_upper_limits[finger_slice],
    )
self.control.joint_target_q.assign(target_q)
```

Keep the existing IK-result and body-transform `.numpy()` calls; they provide
the CPU values required for command validation and GUI TCP errors. Do not add a
Warp synchronization call before either copy.

- [ ] **Step 4: Add direct and captured execution paths**

Add these methods to `Example`:

```python
def capture(self):
    self.graph_ik = None
    self.graph_sim = None
    if not self.model.device.is_cuda:
        return

    with wp.ScopedCapture(device=self.model.device) as capture:
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iterations)
    self.graph_ik = capture.graph

    with wp.ScopedCapture(device=self.model.device) as capture:
        self.simulate()
    self.graph_sim = capture.graph

def simulate(self):
    for _ in range(self.sim_substeps):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
```

Call `self.capture()` once at the end of `__init__`, after the persistent IK,
state, control, and cached host buffers exist. Keep `sim_substeps == 10`, so
capture and every launch preserve the Python state-reference ordering.

Replace the direct IK call in `_update_commands()` with:

```python
if self.graph_ik is not None:
    wp.capture_launch(self.graph_ik)
else:
    self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iterations)
```

Replace the inline simulation loop in `step()` with:

```python
if self.graph_sim is not None:
    wp.capture_launch(self.graph_sim)
else:
    self.simulate()
```

Do not catch CUDA capture failures. The direct path is only the non-CUDA
fallback; a CUDA capture error must remain visible.

- [ ] **Step 5: Run the CUDA integration test and verify GREEN**

Run the Step 2 command. Expected: PASS after 180 frames through both graphs,
including the existing TCP, joint-limit, root-drift, finger, and rate-limit
assertions.

- [ ] **Step 6: Add the user-facing performance changelog entry**

Insert at a random position under `[Unreleased] / Changed`:

```markdown
- Accelerate the Mobile ALOHA example with captured CUDA graphs while preserving its 24-iteration IK and ten-substep dynamics schedule.
```

- [ ] **Step 7: Run focused hooks and commit**

```bash
uvx pre-commit run --files \
  newton/examples/robot/example_robot_mobile_aloha.py \
  newton/tests/test_example_robot_mobile_aloha.py \
  CHANGELOG.md
git diff --check
git add newton/examples/robot/example_robot_mobile_aloha.py \
  newton/tests/test_example_robot_mobile_aloha.py CHANGELOG.md
git commit -m "Accelerate Mobile ALOHA with CUDA graphs"
```

---

### Task 7: Measure performance and visually validate interaction

**Files:**
- Verify: `newton/examples/robot/example_robot_mobile_aloha.py`
- Verify: `newton/tests/test_example_robot_mobile_aloha.py`
- Verify: `newton/tests/test_examples.py`

**Interfaces:**
- Consumes: `Example.graph_ik`, `Example.graph_sim`, and `Example.simulate()`
  from Task 6.
- Produces: measured captured-versus-direct timings and a visually accepted GL
  run without adding a timing-dependent CI assertion.

- [ ] **Step 1: Benchmark graph and direct core execution after warmup**

Run this local CUDA benchmark:

```bash
uv run --extra dev python - <<'PY'
from types import SimpleNamespace
import time
import warp as wp
from newton.examples.robot.example_robot_mobile_aloha import Example
from newton.viewer import ViewerNull


def measure(fn, count=100):
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(count):
        fn()
    wp.synchronize()
    return 1000.0 * (time.perf_counter() - start) / count


with wp.ScopedDevice("cuda:0"):
    example = Example(ViewerNull(num_frames=200), SimpleNamespace(asset_root=None, test=False))
    raw_ik = measure(
        lambda: example.ik_solver.step(
            example.joint_q_ik,
            example.joint_q_ik,
            iterations=example.ik_iterations,
        ),
        30,
    )
    graph_ik = measure(lambda: wp.capture_launch(example.graph_ik))
    raw_sim = measure(example.simulate, 30)
    graph_sim = measure(lambda: wp.capture_launch(example.graph_sim))
    full_frame = measure(example.step, 30)

    print(f"IK raw={raw_ik:.3f} ms graph={graph_ik:.3f} ms speedup={raw_ik / graph_ik:.2f}x")
    print(f"SIM raw={raw_sim:.3f} ms graph={graph_sim:.3f} ms speedup={raw_sim / graph_sim:.2f}x")
    print(f"FULL frame={full_frame:.3f} ms fps={1000.0 / full_frame:.1f}")
PY
```

On the reference RTX 5090, require at least `3x` speedup for each captured core
sequence. The pre-change measurements were `9.857 ms` for IK, `23.840 ms` for
simulation, and about `29 FPS` for a complete headless frame. Do not encode a
wall-clock threshold in `unittest`.

- [ ] **Step 2: Run focused regression tests**

```bash
uv run --extra dev -m newton.tests -p test_example_robot_mobile_aloha.py
uv run --extra dev -m newton.tests -p test_examples.py -k example_robot_mobile_aloha
```

Expected: all helper tests, the 180-frame CUDA integration rollout, and the
registered subprocess example pass.

- [ ] **Step 3: Launch and inspect the optimized interactive viewer**

```bash
uv run --extra examples -m newton.examples robot_mobile_aloha --device cuda:0
```

Verify the steady-state FPS improvement, both draggable TCP gizmos, independent
gripper sliders, decreasing reachable-target errors, fixed base mechanisms,
and no startup jump or visual scale regression. Close the window after visual
acceptance.

- [ ] **Step 4: Verify repository state**

```bash
uvx pre-commit run --files $(git diff --name-only origin/dev...HEAD)
git diff --check
git status --short --branch
git log -12 --oneline --decorate
```

Confirm the pre-existing modified
`docs/images/examples/example_cloth_limx_three_tshirts_box.jpg` and untracked
`solver_convergence.png` remain unstaged and unchanged. Do not push until the
user explicitly requests it.
