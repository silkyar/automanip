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
