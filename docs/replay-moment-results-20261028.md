# Replay moments: stable values did not produce early stacks

2026-10-28. The [registered protocol](replay-moment-study-plan-20261028.md)
followed the passed15,360-transition execution smoke. Three fresh seed1 learners
ran in parallel, with identical initial actors and matched update cadence.
The task, reward, observation/action interface and independent scorer stayed
frozen. This is a CrossQ-inspired vectorized adaptation, not an official CrossQ
reproduction or a demonstrated strong SAC baseline.

## Result and decision

| Arm | Actual transitions | Trainer process seconds | Critic / actor calls | Frozen development |
|---|---:|---:|---:|---|
| Target SAC |874,880|586.449|54,363 /18,121|0/128 at all5 checkpoints|
| Joint moments |596,544|586.448|36,968 /12,322|0/128 at all5 checkpoints|
| Split moments |409,600|586.860|25,283 /8,427|0/128 at all4 checkpoints|

All arms were deadline-censored below the1.024M transition ceiling. Their common
51,200/102,400/256,000 checkpoints are the equal-step comparisons; final results
have unequal data and compute. At256k each has15,688 critic and5,229 actor/alpha
calls. None passes the preregistered10pp/curve promotion gate. No replication or
confirmation is justified from this screen. T80/T90 are unobserved beyond each
arm's actual cutoff, not measured at its requested ceiling. No10× improvement
or general failure of replay is established.

The14 scorer calls completed with128 episodes each and no execution failures.
Behavior training counters also recorded zero terminal stacks. A separate
[source review](replay-screen-independent-review-20261028.md) found no concrete
temperature-sign, loss-broadcasting, reset-successor or export-squash defect
that explains these outcomes away. That review is not a proof of every runtime
detail. In particular, the target-SAC comparator has not demonstrated robot
competence; beating that comparator numerically would not establish a strong
learning result.

## What the saved data distinguishes

The [fixed-policy scalar control probe](research-window-replay-value-20261028.md)
confirmed a structural restriction: warmup split centering immediately before
an affine readout forces each head's current/next batch means to agree. In that
toy this creates gross true-Q error despite a relatively small self-TD loss.
Joint centering avoided the pathology but did not outperform the competent
target-network toy control; its action-derivative RMSE was worse in all3 seeds.
This is specific algebra/toy evidence, not a true-value oracle for StackCube.

Post-hoc CPU inspection of the archived robot snapshots used frozen clones,
numeric-only NPZ loading and declared CPU action noise. It performed no fitting,
simulator or scorer call. At the common256k checkpoint:

| Arm | Recent mean dense reward | Running-Q / batch-context discrepancy RMSE | Running policy action-gradient mean norm |
|---|---:|---:|---:|
| Target SAC |.0522|0|1.4684|
| Joint moments |.0676|1.6552|.2215|
| Split moments |.0427|49.2638|1.4624|

The saved64-episode windows had at most one episode ever grasping in target SAC,
and none grasping in either normalized arm. None ever stacked. The critic's
static flag is not a completed stacking stage: a cube sitting untouched on the
table is static too. Training behavior changes within these windows, so they
are not frozen-policy rollouts or independent success estimates.

Later last-call logs show different numerical failures: target SAC's Q mean
rose to611 with loss35,473, split to−8,086 with loss166,368, while joint stayed
near2.79 with loss1.31. These are sparse last-minibatch statistics, not averages
over the run or proof of true-Q accuracy. On the separately reconstructed final
recent-state context, running-versus-batch Q discrepancy was13.05 for joint and
3,371 for split. That context is recent data rather than the historical uniform
replay batch; discrepancy cannot be equated with a causal calibration error.

A useful necessary consistency check is available without a robot value oracle.
For a **fixed present alpha**, discount.99, reward bounded by1 and continuous
actions in[-1,1]^8, any exact soft Q satisfies

`Q <= 1 + .99*(1 + alpha*8*log(2))/.01`.

This follows from the maximum differential entropy on the bounded action box.
The pinned official normalized reward source was retrieved and hash-matched to
the task before applying this check. For final target SAC, present alpha.42354
gives upper bound332.51;30% of the saved recent states' deterministic-policy
min-Q predictions exceed it, with mean915.36. Thus those predictions are not
accurate fixed-present-temperature soft values. Alpha changed during learning,
so this does not reconstruct historical targets or prove a programming bug.
The bound is one-sided and does not diagnose split's negative values.

Stable joint values therefore do not establish a useful learning mechanism.
The remaining bottleneck includes learning action distinctions that cause
reaching/grasping, rather than merely making a self-bootstrap numerically stable.
The current observations do not tell us whether actor-gradient bias, delayed
credit, coverage or continuing-return/finite-horizon mismatch dominates.

## Preservation, accounting and limits

- Archive: `runs/stackcube-replay-moments-screen-20261028/`; all86 worker files
  hash-verified, plus trusted summary,14 reports and exact source/protocol copies.
- Three scientific runs used1,881,024 new transitions. The separate smoke used
  15,360; the public publication-video render used400. None used teachers,
  demonstrations or extra planning queries for training.
- Learner sampled rows:111,335,424 /75,710,464 /51,779,584 for target/joint/split;
  these are replay processing, not new environment transitions. Actor/alpha
  updates and forward counters are retained separately in the trusted summary.
- All normalization stayed below100,000 critic updates: this was warmup BN,
  not postwarm BRN. The adapted UTD1/16 and batch2048 differ from the paper.
- Final target/joint50-step windows cross episode boundaries. Only aligned
  snapshots can be treated as complete episodes; no fabricated final Monte
  Carlo return is used. Saved states omit target networks, optimizer and full
  replay, so exact historical target-SAC updates cannot be reconstructed.
- Snapshot diagnostic source: `scripts/diagnose_replay_screen.py`; receipt:
  `research/discovery/replay-screen-snapshot-diagnostics-20261028.json`.
  Runtime1.5064 CPU seconds,44,800 previously collected transition rows
  inspected,0 optimizer calls/new interactions/scorer calls. It measures
  gradients' scales, not their correctness.
- All three associated Modal apps were verified stopped with zero tasks.
  Provider-reported shared-session spend at17:30UTC was$13.50352218, including
  earlier research; remaining$86.49647782 under the$100 cap. Metering can lag.

The next step is a discriminating cheap measurement of why a stable critic
still fails to support action learning, not another three-seed extension or an
unrelated algorithm sweep.
