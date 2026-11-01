# Upstream verification

Inspected 2026-10-19. Network access from the shell was unavailable; source was
read through the web tool. No simulator was installed or executed locally.

## Task and metric

[Current PickCube source](https://raw.githubusercontent.com/haosulab/ManiSkill/main/mani_skill/envs/tasks/tabletop/pick_cube.py)
registers `PickCube-v1` with a 50-step limit. For Panda, stock success combines
object-goal distance at most 0.025 m with a static arm. Grasp/release is not a
success prerequisite. Goals may be above the table, so this is not a placement
and release benchmark. State observations include privileged object/goal features.
The implementation here reproduces the stock thresholds as a pure scalar rule;
it is not yet connected to simulator-owned state.

## Controller

[Pinned v3.0.0b21 Panda source](https://raw.githubusercontent.com/haosulab/ManiSkill/v3.0.0b21/mani_skill/agents/robots/panda/panda.py)
exposes `pd_ee_delta_pose`, combining end-effector translation/rotation deltas
with a mimic gripper. The intended normalized action interface has seven values;
its actual Gym space remains to be asserted in an integration test.
`Panda.is_static(0.2)` takes the largest absolute arm-joint velocity, excluding
both finger joints. It is not a velocity vector norm.

## Reset, step, and devices

[Pinned BaseEnv source](https://raw.githubusercontent.com/haosulab/ManiSkill/v3.0.0b21/mani_skill/envs/sapien_env.py)
accepts `reset(seed=int_or_list, options=...)` and returns observation/info.
Partial reset uses `options["env_idx"]`; reconfiguration cannot accompany a
partial reset. `step(action)` returns observation, reward, termination, truncation,
and info. BaseEnv termination follows task success/failure; its own truncation
array is false, so wrappers supply time limits. CPU simulation requires one
environment (`physx_cpu`); GPU simulation uses `physx_cuda`. Renderer support is
separate from physics support. This does not establish that this Mac supports
a complete simulator installation.

## PPO reference and deliberate changes

[Current official PPO](https://raw.githubusercontent.com/haosulab/ManiSkill/main/examples/baselines/ppo/ppo.py)
uses state observations, a three-layer 256-unit tanh actor/critic, and random
initialization unless given a checkpoint. Its default controller is
`pd_joint_delta_pos`, unlike our requested end-effector mode. Its environment
configuration explicitly requests CUDA physics, even when policy device selection
falls back to CPU. Evaluation uses `ManiSkillVectorEnv`, ignores success termination
by default, and collects wrapper episode metrics. A future evaluator must label
success-at-end versus success-ever explicitly. We will not copy its pickle-based
checkpoint loading into the trusted artifact interface.

## Pinning status and remaining checks

`mani_skill==3.0.0b21` is the provisional simulator pin. The pinned Panda and
BaseEnv files were readable; requests for pinned PickCube and PPO files failed.
Task/PPO findings above therefore refer to current `main`, not a claim of exact
release equivalence. Before simulator integration: retrieve the pinned task and
baseline, compare source, resolve and lock transitive dependencies on the target
Linux platform, assert spaces and flattening order, and exercise seeded resets.
Do not promote this recipe as reproducible training until those checks pass.

## Local installation update

Network access became available through approved shell execution. Installed the
pinned ManiSkill release and inspected its installed task/controller source.
Created project-local Python 3.11.17 with uv. The old SAPIEN guide URL returned
404; used the official nightly macOS asset 3.0.4.dev20261009+7dba23f1 instead.
Pinned OpenCV 4.11.0.86 to retain NumPy 1.26.4. Installed `pin==2.7.0` for CPU
inverse kinematics and MoltenVK v1.4.2 for Vulkan initialization.

Dependency consistency checks pass. PickCube CPU initialization, one zero-action
step, and repeated seeded reset succeeded with 42 observation and 7 action values.
The observation size is now checked against the frozen task specification.
No GPU physics, policy training, reference performance, or vectorized real-device
throughput has been verified. The macOS snapshot is not a tested Linux lock.

## PPO source retrieved and adapted

With approved network access, the exact v3.0.0b21 official PPO source was fetched
successfully and used to implement `train.py`. Installed pinned-release task and
vector-wrapper source was also inspected. Final-observation masks and both
termination/truncation paths are handled explicitly. The 50-step CPU training
wrapper check passed without any optimizer updates. See `docs/training.md` for
the recipe, attribution, intentional changes, and remaining GPU validation.

## Linux GPU verification update — 2026-10-25

The hashed Linux lock installed on an official Runpod Python 3.11 image and
passed dependency checks and all 31 CPU tests. CUDA PhysX wrapper checks and
three random-init training/development-evaluation runs passed on an NVIDIA L4.
The frozen 42-observation/seven-action interface was checked on this backend.
This dated result supersedes the earlier GPU-pending status above; the source
and controller differences remain. See `docs/runpod-reference-20261025.md`.
