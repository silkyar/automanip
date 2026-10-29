# AutoManip: autonomous research for simulated robotic picking

## Motivation
Inspired by Andrej Karpathy's autoresearch (<https://github.com/karpathy/autoresearch>) and the minimalist training philosophy of nanoGPT/nanochat. Investigate how much a coding/research agent can improve a small robotics policy through repeated, bounded, hypothesis-driven experiments without human selection of each change.

## Initial task and stack
- Simulator: ManiSkill 3 (verify current upstream APIs before implementation).
- Starting environment: `PickCube-v1` with a Franka Panda arm.
- Observations: privileged state (not images in v0).
- Control: stable, pinned end-effector delta-pose controller; confirm exact ManiSkill control-mode identifier from upstream code.
- Algorithm: state-based PPO, adapting the official ManiSkill baseline.
- Initialization: random weights under a reproducible training recipe. Train the official reference recipe, evaluate it, and retain its checkpoint as a reference. Do not require pretrained weights for v0.
- Target experimental training budget: about 10 minutes per experiment on an eventual NVIDIA cloud GPU, subject to benchmarking. Record environment steps and wall time; make fair comparisons under a fixed budget.
- Do not assume a working GPU or simulator installation yet. Begin locally with code, tests, and scaffolding.

## Metrics and task definition
Primary v0 benchmark: official ManiSkill PickCube success metric to compare with official PPO results, measured from evaluator-controlled simulator state. Its ordinary success criterion is cube near target and robot sufficiently stationary; it does **not** necessarily require releasing a cube.

Secondary strict-placement metric: custom table-supported target, object deposited in target region, gripper no longer supporting/contacting the object, object stable for a settling interval. Define and test this independently in an evaluator-owned task. Don't silently apply strict placement to the official PickCube environment if target semantics differ.

Track success, grasp rate, lift rate, target distance, episode length, action smoothness, failure/invalid-action counts, and optionally physics stress/generalization metrics. Timeouts and crashes count as failures.

## Threat model / non-cheating
The autonomous agent controls training experiment code but not trusted evaluator, private seeds, success rules, simulator dependency versions, or artifact gatekeeper. Separate the trainer and evaluator; the latter accepts only a restricted policy architecture/config manifest plus safe tensor weights. It must not execute arbitrary training-supplied code or trust rewards/success flags supplied by the candidate.

A single local repository cannot provide adversarial security if the agent has unrestricted shell access. For unattended autonomous runs: use separate containers/VMs/service accounts with the evaluation code, simulator state, holdout seeds, and scorer inaccessible and read-only/remote as appropriate. No eval credentials or private seeds in the trainer mount, logs, stack traces, or environment variables. Note trade-offs clearly in local MVP.

Development episodes can be fixed for paired comparisons; this set will become adaptively overfit over many trials. Confirm promising improvements with fresh independent episode seeds, several independent training seeds, and a final private holdout that is not repeatedly accessed. Proposed initial counts: 128 development episodes, 512 confirmation episodes per candidate, and 1000 final holdout episodes, adjustable after throughput calibration.

## Editable vs immutable
Editable by research agent, initially constrained to `train.py` and explicit training-config inputs:
- PPO optimization settings, network architecture within restricted policy schema, entropy/exploration schedules.
- Training-only reward shaping and curriculum parameters; immutable evaluation reward and task success logic.
- Training schedules and minibatching, provided compute and training budget are recorded.

Immutable by research agent:
- `evaluate.py`, evaluation task definition, evaluation seeds and holdout, episode horizon, controller and observation specs, success checker, validator, benchmarking scripts and dependency locks.

## Reproducibility
Every result should record: git commit, hypothesis, code diff or experiment identifier, simulator and software versions, device, training wall time, environment transitions, learner updates, random seeds, checkpoint digest, development and confirmation metrics, errors, and decision to keep/revert. Add matched-seed comparisons and uncertainty reporting. Budget is measured separately from eval runtime.

## Suggested architecture (adapt as needed)
- `AGENTS.md` — collaboration and safety instructions for Codex.
- `PROJECT_SPEC.md` — this design.
- `pyproject.toml` — pinned dependencies and test extras.
- `src/automanip/task_spec.py` — frozen observation/action/task spec.
- `src/automanip/policy_io.py` — policy manifest/weight validation and safe loading.
- `src/automanip/evaluator.py` — trusted policy rollout and scoring.
- `src/automanip/metrics.py` — benchmark and strict placement check.
- `train.py` — eventually agent-editable training entry point.
- `scripts/run_baseline.py` — run reproducible reference training on GPU later.
- `scripts/evaluate.py` — controlled evaluation entry point.
- `tests/` — CPU runnable tests for policy interfaces, scoring, artifacts, mock rollouts.
- `results/` — experiment history; checkpoints stored separately.

## Implementation milestones
1. Inspect upstream ManiSkill `PickCube-v1`, reset APIs, controller mode, success conditions, PPO script, and device requirements. Pin compatible versions. Summarize factual differences versus our assumptions.
2. Scaffold project, create a typed/pinned task config, safe policy artifact format, and CPU-running unit tests for scoring and validation (no cloud GPU required).
3. Implement an evaluator able to handle ManiSkill vectorized environments and correctly distinguish termination, truncation, and success. Add a deterministic smoke test where supported.
4. Port/adapt official PPO baseline to an editable minimal `train.py`. On cloud GPU, train 3+ independent seeds and assess reference performance; record checkpoint manifest and all metrics.
5. Add constrained training reward and curriculum hooks and an isolated experiment runner. Enforce training/evaluation permission separation.
6. Add autoresearch-style experiment history, hypothesis loop, automatic Git keep/revert, confirmed candidate promotion, and spending limits.
7. Extend toward strict table placement, RGB + proprioception, and later held-out physics / object distributions.

## First Codex assignment
Please inspect `AGENTS.md` and `PROJECT_SPEC.md`, verify the relevant current ManiSkill implementation and PPO baseline, and implement only steps 1–2 initially. Show me the code, explain the exact APIs you verified, and ensure CPU-only unit tests run. Do not assume a GPU is available, and do not create an insecure evaluator that imports arbitrary candidate training code.
