# Grow–Plan–Distill pilot: closed negative, October 31

The bounded pilot did not produce a useful policy improvement. The historical tuned PPO result remains **95.9% independent confirmation success**. These experiments adapted two much weaker acquired 8.192M-step actors; their scores are not replacements for that benchmark or evidence of faster learning from scratch.

The most useful discovery concerned action coordinates. Panda arm commands are relative joint movements, but its gripper command is an absolute position target. A small command around zero therefore closes the gripper rather than holding its current opening. Changing only the proposal coordinate to a current-relative finger target increased best-of-bank successful continuations from **1/128 to 89/128 source states**. This is training-time search coverage, not a deployable success rate: all attempted branches and replayed prefixes count toward experience.

![Oracle coverage and two separate fresh-policy comparisons](../artifacts/cumulative/gpd-pilot-closeout-20261031/mechanism-summary.png)

The oracle and policy panels use different vertical scales and different episode sets; they are separate experiments, not one learning curve.

## What survived closed-loop testing

| Fixed comparison | Original/control | Intervention | Conclusion |
|---|---:|---:|---|
| First readout student, 1,024 fresh episodes per method | Original actors 1; equal-physics unbranched student 14 | Search-data student 34 | Small improvement; failed predeclared +10 percentage point learning and +5 point search-specific screens. |
| Remove action rehearsal, 512 fresh episodes per method | Rehearsed student 15 successful stacks | Teacher-only student 0 | Removing rehearsal destroyed grasping: only 16/512 ever grasped, versus 508/512 with rehearsal. |
| Preserve executed rather than raw actions, 1,024 fresh episodes per method | Closed form 18; matched raw refinement 17 | Executed-action refinement 37 | Positive small effect, preserved grasping; failed +10 point material screens against both controls. |
| One data aggregation step, 512 fresh episodes per method | Fixed student 12; extra original-policy states 6 | Extra student-policy states 6 | No benefit from student-state support in this fixed comparison. Grasp retention passed. |

All rows use fresh public episodes for their respective comparison. Intervals condition on only two acquired parent policies and resample paired 64-episode simulator blocks; they are not uncertainty estimates over a broad training-seed population. No confirmation or final holdout was queried.

For the final aggregation comparison, student-state minus baseline success was **−1.17 percentage points**, with a paired block 97.5% interval **[−2.73, +0.20]**. Student-state minus original-state aggregation was **0.00 points [−1.17, +1.17]**. Each collection bank had successful labels for 43/128 states. Both fits used 192 fixed old teacher rows, 192 new rows, and the same 1,024 rehearsal rows, frozen features and loss. The compatibility scorer role names are explicitly mapped in the [machine result](../cumulative/results/onpolicy-support-score-20261031.json).

## Why more variants are not justified here

Finding an occasional successful continuation is much easier than representing and executing a state-dependent correction without damaging earlier behavior. Held-out finger-target errors remained roughly 9–14 mm in the initial readout, while many successful teacher increments were below 1 mm. Whole-network fitting, an auxiliary finger-coordinate head, and function-preserving input normalization did not pass the prespecified precision gate; they were not scored on the robot. A separate spectral diagnostic found that removing the fixed ridge penalty analytically did not close this precision gap. These observations concern these learners and data, not an impossibility result for distillation or learned representations.

The rehearsal endpoint establishes a narrower causal result: preserving early behavior was necessary in this local learner. It does not establish that the chosen rehearsal weight was optimal. Executed-action preservation reduced one unnecessary constraint and gave a small gain, but did not solve the control problem. One aggregation step did not solve it either; this was not a complete implementation or test of DAgger.

An additional validity result matters for future planning: writing captured simulator state back perturbed subsequent contact dynamics even when immediate observed state matched. Fresh subprocesses replaying identical prefixes were reproducible in our test, so later branches used that more expensive method. The exact simulator cause remains unknown. No state-write result was treated as a valid counterfactual intervention.

## Accounting and disposition

The pilot used **984,320 of its 1,000,000-transition ceiling**, including validity failures, repeat checks, all branches, control collection and fresh scoring. It is now closed; the remaining 15,680 transitions will not be spent on score-driven refits. CPU work included 26 linear solves, four SVDs and 4,096 gradient updates including 1,024 updates from four fits that failed at post-training validation. Failed artifacts and logs are retained. CPU optimizer work is separate from the historical **84 GPU training runs across 44 families**.

The last app is archived and read back as stopped with zero tasks. All **1,684 protected files** match the pre-existing baseline. Sources, safe policy weights, fit row indices, simulator reports and failures are retained under `artifacts/cumulative/` and `.runtime/`; the [accounting record](../cumulative/results/gpd-pilot-closeout-20261031.json) lists all transition categories. No task, evaluator, dependency pin, historical benchmark or success rule was changed.

The next opportunity is to test the action-coordinate finding directly in learning from random initialization, before investing in a larger planner or distiller. First test a known gripper hold mapping inside the existing safe MLP on CPU, with an exactly matched capacity control. Such an initialization would be an explicitly disclosed actuator prior. Whether it improves exploration or merely delays grasp acquisition is an open empirical question.
