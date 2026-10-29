# StackCube quality gap: three bounded PPO experiments

The user prioritizes reliable near-100% StackCube success before limiting training
time and explicitly asks for experiments to remain ready while jobs finish.
This overrides the earlier priority on 10-minute/sample-efficiency performance
for the next batch. The current confirmed full-horizon reference remains 94.08%
across three training seeds. PickCube and all original StackCube results remain
unchanged.

The latest CAPG variant failed its predeclared development screen. The adapted
SAC control scored 0/128 at all four measured checkpoints, including its final
262,432-transition policy after 1,292.19 trainer seconds. Neither is promoted.
These are recipe-level results, not universal algorithm comparisons.

## Hypotheses and matched comparison

Three fresh seed-1 full-horizon PPO runs share gamma .9, initial entropy .003,
initial learning rate .0003, eight update epochs, 1,024 environments, 50-step
rollouts, 32 minibatches, and the current actor/critic architecture. Each requests
50M transitions (49,971,200 after rollout alignment), with a 2,700-second trainer
cap and a 20-second save reserve. All source changes are narrow training-side
schedule/audit additions; the task, reward, physics, observations/actions,
evaluation horizon, success rule and safe artifact contract stay fixed.

| Arm | Through 25.6M transitions | Later refinement |
|---|---|---|
| precision-constant | Current strong recipe | Entropy .003 and LR .0003 stay constant |
| precision-entropy-anneal | Same recipe | Entropy linearly falls to zero by 49,971,200; LR stays .0003 |
| precision-lr-anneal | Same recipe | LR linearly falls to .00003 by 49,971,200; entropy stays .003 |

The constant arm asks whether the gap is simply additional training. Any such
gain is credited to the larger training budget, not a new learning mechanism.
The other arms test whether reducing continued exploration or update magnitude
after a competent policy emerges improves final reliability. Previous global
entropy decay used early success resets and a different schedule; its failure
does not establish that this full-horizon, later refinement will work. It remains
negative evidence, and the new matched constant arm is essential.

The public stock reward already includes ungrasping and static-placement rewards;
this batch does not assume either term is missing. Existing development failures
motivate care around release and subsequent stability, but do not establish causes.
No demonstrations, pretrained models, teacher labels or planning data are added.

## Fixed budget and scoring

- At most three dedicated L40S trainers in parallel, each from random seed 1.
- Original absolute 2,700-second lease fixed by the trusted dispatcher, never
  refreshed after queueing, failure or preemption; external hard stop mandatory.
- Wrapper ceiling 2,760 seconds plus a 30-second startup allowance. No GPU packing
  or other utilization experiments.
- Frozen 128-episode development scoring at existing saved checkpoints 25.6M,
  38.4M and the final actual-step checkpoint: at most nine evaluations total.
  Missing later points are censored; identical step/weight artifacts are deduped.
- Preserve every native checkpoint/event/log, safe actor, source hash, actual
  transitions, trainer/evaluator time, exact optimizer steps, configured entropy
  and all eight learned log standard deviations. Built-in evaluation transitions
  are separate from training and frozen development experience.
- Initial total cap $7. Conservative resource bound is approximately $6.16 at
  the current documented rates; actual meter and any lag are reported afterward.
- No additional seeds or confirmation queries in this admission. No silent reruns.

## Decision before viewing outcomes

The working performance target is at least 99% final success across multiple
training seeds and independently scored episodes. This initial single-seed screen
cannot establish that milestone. A final development count of at least 126/128
(98.44%) is sufficient to propose a separately budgeted multi-seed replication.
If the constant arm reaches that level, prefer it unless a refinement arm improves
by at least three paired development successes. If only a refinement reaches the
screening level, replicate it with the constant comparator before promotion.
Ties default to the simpler constant recipe. Inspect complete learning curves
and failure diagnostics; no checkpoint cherry-picking or confirmation feedback
may select the winner. Confirmation remains reserved for promotion, and no private
final holdout claim is made.

If all arms remain below the screening level, record the negative result and use
the phase-conditioned diagnostics to choose the next mechanism. Additional compute
is reported explicitly: these results are not equal-budget comparisons with the
25M reference. Once reliability improves, training-time reduction becomes the
next objective.

Implementation: `src/automanip/precision.py`. Three CPU tests verify exact schedule
boundaries, monotonic isolated parameter changes, and byte-for-byte recovery of
the reviewed base trainer after removing marked additions. The optimizer counter
increments only after an actual optimizer step, including KL early-stop cases.
Payloads and exact hashes are in `.runtime/modal-precision-payloads/manifest.json`.
