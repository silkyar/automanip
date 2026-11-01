# Adapted SAC control screen — 2026-10-27

One seed-1 learning run, from random initialization, to establish a conventional
off-policy comparator. The user authorized continued learning experiments rather
than further GPU utilization work. This is not a broad sweep or a novel algorithm.

## Readiness and preserved failure

The candidate reuses the pinned ManiSkill SAC design with declared adaptations:
two tanh actor hidden layers to fit the existing safe policy schema, gamma .9,
stable squashed-Gaussian likelihoods and explicit compute accounting. Details and
upstream source hashes are in `research-discovery-sac-baseline-20261027.md`.
Neither the official README's PickCube examples nor a CPU toy establish StackCube
competence. The current strong reference is independently confirmed full-horizon
PPO, not the original weaker official PPO recipe.

The first attempted smoke failed at the replay dtype boundary after its first
eight simulator transitions, before any learner update. Its source, logs, claims
and output Volume are preserved. A regression test now inserts the actual boolean
terminal mask into float replay without changing truncation/terminal semantics.
All 17 discovery tests pass. Corrected trainer SHA256:
`1191b6bafb1c5a34cb2f661a23da732cf2021ec680f3971477d4c02fa8f614ca`.

The corrected-version GPU smoke passed in 11.04 trainer seconds: 800 transitions,
100 vector steps, 16 completed episodes, 24 reset episodes including initialization,
and 93 joint-critic, actor and temperature updates. All four exported actor files
passed the unchanged trusted loader and produced finite bounded actions. This is
an execution/interface check, not robotic policy success. Reported total cost of
the SAC smoke admissions was $0.02274, with possible billing lag. No confirmation
or private holdout was queried.

## Fixed recipe and budget

- One dedicated L40S, random training seed 1.
- Official StackCube-v1 normalized dense reward, panda_wristcam, privileged
  state48/action8, pd_joint_delta_pos, horizon 50, no early success reset.
- 32 environments, replay capacity 500,000, minibatch 1,024, 4,000 random warmup
  transitions, 64-transition collection blocks, .5 SAC update rounds per
  transition, gamma .9, tau .01 and learning rate .0003.
- Ceiling 500,000 training transitions or 1,320 trainer seconds, whichever comes
  first. Include child startup and export in the trainer budget. Retain the
  20-second save reserve and external hard stop. No automatic fresh lease,
  restart or extra data after failure/preemption.
- Export every 51,200 transitions and finally. Preserve every export and raw log.
- Frozen 128-episode development scoring at 51,200, 102,400, 256,000 and the final
  actual-step actor. Absent later points are censored, not failed hypothetical
  policies. Dedupe coincident step/weight artifacts without repeating evaluation.
- Maximum four development evaluations. Separate training from evaluation steps,
  runtime, setup and queue wait. Overall budget $1.50, including those evaluations
  and overhead, separate from the earlier smoke and paired PPO screen budgets.
- Trainer receives no evaluator, results, credentials, demonstrations, teacher,
  pretrained weights or planning rollouts. Trusted scoring loads numeric actor
  artifacts only; it never imports candidate source. Evaluation stays unchanged.

## What this screen can establish

Report the full learning curve, actual transitions, reset episodes, critic/actor/
temperature updates, replay minibatch rows, alpha/loss traces, per-stage development
outcomes and wall-clock components. Each joint critic update trains two Q networks.
Reset time is already included in environment step timing and must not be added
again. Export cost, failed-smoke experience and all evaluation experience remain
visible in accounting.

Record first sampled 80/85/90% crossing brackets if reached; otherwise say not
reached within the measured budget. Do not extrapolate a time to success. The
checkpoint cadence differs from PPO and their final transition budgets differ;
show those domains explicitly. This compares two complete recipes, including
different actor structures, rather than isolating replay as a causal mechanism.

A policy reaching at least 80% final development success is worth a separately
planned multi-seed screen. A lower score is diagnostic, not a refutation of SAC
or replay: inspect learning progress and failure phases before choosing a next
experiment. Never call a single adapted seed a discovery, a reliable gain or a
10x result. No confirmation queries or extra seeds are included in this admission.
