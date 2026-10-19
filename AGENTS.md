# AutoManip — Codex repository instructions

Read `PROJECT_SPEC.md` before making architectural decisions. The user is building a Karpathy-autoresearch-style autonomous experiment loop for small robot manipulation tasks.

## Current phase
Implement and review code locally, without requiring a cloud GPU yet. Do not claim GPU tests have been run. Start by inspecting upstream ManiSkill APIs and the official state-based PPO baseline; pin dependency versions or commits when implementing.

## Principles
- Prefer a minimal, understandable codebase with clearly bounded research-agent edit permissions.
- Begin with ManiSkill PickCube and state-based PPO, from a reproducible random initialization, not arbitrary pretrained weights. Save a reference checkpoint once trained.
- Treat evaluation code, benchmark distributions, seeds, and policy artifact validation as trusted, frozen components. Never allow the autonomous research agent to edit these.
- Differentiate ManiSkill stock PickCube success from true pick/place/release success. Do not call the stock metric strict placement success.
- A frozen evaluator file is not by itself a security boundary. Eventually run trainers and evaluators in genuinely isolated environments with separate filesystem privileges; do not claim a local single-user MVP prevents a malicious agent from cheating.
- Keep training reward and curriculum editable only through narrow, explicit interfaces. Fix observation and action interfaces for version 0.
- Avoid untrusted pickle deserialization and arbitrary candidate Python execution inside the trusted evaluator. Use restricted policy architecture metadata and safetensors or equivalently safe tensor serialization with validation.
- Validate candidates with paired development episodes, independent confirmation seeds, repeated training seeds for promising variants, and a private final holdout; always log compute budget, environment steps, and training seeds.
- Use strong test isolation and unit tests for evaluator rules, manifests, and candidate artifacts.
- Before doing large implementation work, give a concise plan and explain any assumptions.

## First task
Inspect this spec, propose a small initial repository layout, and implement an MVP that can execute sanity tests on CPU where possible. Build the evaluator interfaces and tests before adding the autonomous research agent. Keep cloud-GPU training as a later milestone.
