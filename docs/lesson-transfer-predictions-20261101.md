# Lesson-transfer test: predictions for an unseen task (registered before any run on it)

Registered 2026-11-01 before any training or evaluation on PokeCube-v1; nothing in this work has run on it. The predictions below are derived only from the mechanism ledger in [sample-efficiency results](sample-efficiency-results-20261031.md). The test asks whether diagnosed mechanisms predict which methods help on a new task, instead of re-running the StackCube search.

**Task analysis (from the task source, not from runs).** PokeCube-v1 (Panda, 50 steps). The robot must grasp a peg and keep holding it (no release step). It must then align the peg head with the cube to within 0.05 rad yaw and cube half-size + 5 mm, and push the cube into a 5 cm goal radius. The relevant ledger mechanisms:

| ledger mechanism | evidence | present in PokeCube? |
|---|---|---|
| release plateau from unbounded gripper certainty (fixed by the ε-floored gripper) | StackCube, PlaceSphere; absent and no gain on LiftPegUpright (a1) | **No** (the peg is held throughout) |
| update cadence (512 vs 1024 envs) helps generally | StackCube 0.57×, LiftPegUpright 0.60× | yes (generic) |
| high-UTD TD3 helps on grasp/transport tasks | StackCube 0.35×, PickCube 0.50×, LiftPegUpright 0.60× | yes |
| rare precision frontier stalls TD3 (5 mm tolerance with nearly flat shaping) | PlaceSphere d1 (all seeds censored) | **Yes** (5 mm and 0.05 rad alignment) |

**Arms (seeds 11–13, research-side evaluation on 128 seeds from 300,000,000, 512k checkpoints).** `ppo-control` (1024 envs, 15.36M), `ppo-env512` (15.36M), `ppo-env512-bineps` (15.36M), `td3-utd16` (7.68M).

**Predictions (T80 = first sustained crossing of 103/128; ratio = median T80 / control median T80, censored at run length):**

- P1. `ppo-env512` beats control: ratio ≤0.80.
- P2. The gripper floor adds nothing on top of cadence: `ppo-env512-bineps` median T80 is within one 1.024M checkpoint of `ppo-env512` (no release step).
- P3. `td3-utd16` does **not** show its usual grasp-task gain. Its median T80 is ≥0.8× control, or it is censored on ≥2/3 seeds, because the alignment condition is a precision frontier like PlaceSphere's. This is the risky prediction: on PickCube and LiftPegUpright TD3 gave 0.50–0.60×.
- P4. If TD3 does stall, the stall is at alignment: peg grasped in ≥120/128 episodes by 1M with low success, rather than at grasp.

**Scoring.** Each prediction is scored right or wrong from the outcome. All four are reported regardless. If P3 is wrong (TD3 gets its usual gain), that is evidence the PlaceSphere stall is specific to that task's reward shaping rather than a category "precision frontier".
