# Project Lessons

## 2026-07-29 — Keep the primary checkout aligned with the user's daily branch

- Context: Configuring this Newton fork as the editable source for Isaac Sim 6.0.1.
- Mistake: Kept `main` at `/home/limx/github/newton` and placed the actual development branch in a hidden `.worktrees/` path, adding unnecessary friction to the user's normal solver workflow.
- Rule: When the user only needs one compatibility branch for daily development, make the canonical repository path check out that branch. Keep other branches available remotely or as Git refs unless the user explicitly asks for simultaneous local checkouts.

## 2026-07-29 — Distinguish integration mechanics from solver design

- Context: The user asked how CUDA code is embedded into Newton.
- Mistake: Redirected the discussion into choosing solver algorithms and joint scope instead of explaining the concrete CUDA/Warp integration boundary.
- Rule: When asked how native or GPU code plugs into an existing framework, first explain the exact module, kernel, launch, state, and export path. Discuss numerical-method design only if the user asks for it afterward.

## 2026-07-29 — Separate vector atomic execution from atomicity guarantees

- Context: Discussing CUDA `atomicAdd(float4*)` on RTX 50-series hardware versus Warp's component-wise vector atomic implementation.
- Mistake: Asserted equivalence from component-wise semantics without first verifying the architecture-specific CUDA implementation and documented guarantees.
- Rule: For GPU intrinsics, distinguish API semantics, compiler lowering, generated instructions, and hardware transaction width. Verify the installed toolkit and NVIDIA's architecture documentation before claiming two implementations are equivalent.

## 2026-07-29 — Preserve the requested Isaac Sim GUI context

- Context: The user asked whether there was a cloth demo after discussing Isaac Sim integration.
- Mistake: Launched Newton's standalone OpenGL cloth example instead of checking for a demo inside the full Isaac Sim GUI.
- Rule: When the active workflow is Isaac Sim and the user asks to run or show a demo, default to the Isaac Sim GUI context. Clearly distinguish standalone Newton examples from Isaac Sim-integrated examples before launching anything.

## 2026-08-05 — Distinguish proposed LIMX contact from Newton's existing rigid-soft path

- Context: Choosing a collision method for a Franka gripper interacting with LIMX cloth.
- Mistake: Recommended an IPC-style rigid proxy surface formulation without first stating that Newton's existing rigid-soft collision pipeline uses analytic/SDF-style shape distance queries and contact records.
- Rule: Before proposing rigid-soft contact for LIMX, inspect and describe Newton's current `CollisionPipeline` shape-query path first. Clearly label any IPC VF/EE extension as a new design rather than existing Newton behavior.

## 2026-08-05 — Preserve the requested VF/EE rigid-cloth direction

- Context: Discussing how the Franka mesh should collide with LIMX cloth after the user explicitly chose vertex-face and edge-edge contact.
- Mistake: Replaced the requested VF/EE design with Newton's existing SDF rigid-soft path instead of answering whether reusable open-source VF/EE code had been found.
- Rule: Treat an explicitly selected collision representation as a fixed requirement. Investigate and cite its concrete implementation source before recommending a different representation; distinguish existing local code, externally reusable code, and a proposed extension.

## 2026-08-05 — Treat LIMX contact as a complete second-order operator

- Context: Designing kinematic Franka triangle-mesh contact for LIMX cloth using VF and EE stencils.
- Mistake: Described applying contact force to the cloth without explicitly requiring the corresponding Hessian operations used by LIMX's Newton solve.
- Rule: Every new LIMX contact constraint must define consistent force, matrix-free Hessian-vector product, and diagonal Hessian blocks. For a kinematic collider, differentiate only with respect to cloth unknowns while retaining the rigid stencil geometry in the residual and derivatives.
