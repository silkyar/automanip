# Proposed StackCube research search

The official 25M-step baseline averaged **32.2% confirmation success** across
three training seeds, with a 22.5–47.7% range. See the
[completed diagnostic report](stackcube-results-20261026.md). These are proposed
experiments; autonomous optimization has not started.

## Reward interpretation

The stock reward already encourages reaching, moving a grasped cube toward the
3D stacking pose, opening once on top, and low cube velocity. Its branches
overwrite one another: reaching yields up to 2, grasping yields `4 + proximity`,
on-top geometry yields `6 + (opening + stillness)/2`, and success yields 8;
the selected value is divided by eight. There is no separate lift-height term.
Every reconstructed reward was checked against the unchanged task method.

In development replays, seed 1's mean normalized gripper command during held
placement was −0.47, versus 0.01 and 0.09 for seeds 2 and 3. Higher commands
request a wider gripper. These step-weighted observations support investigating
release behavior, but do not prove that increasing the existing opening reward
will solve it. Velocity failures also require distinguishing real motion,
recontact and simulation jitter; the videos alone cannot establish the cause.

Proposed objective: **at least 60% average success across training seeds within
the same transition budget**, retaining the full evaluation environment.

1. **Budget control:** run the unchanged recipe longer, separately, to test
   whether the rising final curves mainly reflect undertraining. Extra steps
   cannot count as an improvement at the original budget.
2. **Credit assignment:** test PPO discount 0.9 and 0.95 against 0.8, one change
   at a time, to test whether delayed release/stability outcomes are undervalued.
3. **Lift curriculum:** test a bounded, training-only clearance incentive that
   decays away. Check for holding-at-height behavior rather than assuming lift
   reward automatically improves stacking.
4. **Placement and settling:** test centered alignment and low-motion shaping
   before release. Consider post-release quiet behavior after inspecting contact
   and action traces. Avoid an unconditional gripper-opening bonus.
5. **Exploration:** test entropy coefficients 0.001 and 0.003 against zero,
   or a separately tested noise schedule. Keep architecture fixed initially.
6. **Efficiency:** profile the rollout path first: official logged rollout time
   accounts for about 82% of process time. Evaluate batching and compiled PPO as
   a separate arm, recording algorithm/configuration differences explicitly.

Screen on paired development episodes; repeat promising candidates across three
paired training seeds. Reserve fresh confirmation episodes for promotion and a
private final holdout for the final decision. Current confirmation results are
historical baseline measurements, not a set to query after every trial.
Use separate fixed-transition and fixed-wall-time comparisons; matching one
budget does not automatically match the other. Keep the 48-state/eight-action
interface, controller, 50-step evaluator horizon and scoring rules frozen.
Trainer/evaluator filesystem and identity isolation remains required before
claiming adversarial protection for an autonomous agent.

