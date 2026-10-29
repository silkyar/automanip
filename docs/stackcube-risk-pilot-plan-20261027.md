# Short continuation pilot: episodic lower-tail objective

This is a three-arm diagnostic of a different training objective, running
alongside the longer PPO quality batch. The user prioritizes near-100% StackCube
success and continuous prepared experiments. Total pilot cap: **$1**.

## Evidence and scope

A learned lower-tail critic improved reliability in a constructed CPU toy versus
its mean-return and conventional mean-minus-standard-deviation controls. Expected
reward decreased, making the risk/reward tradeoff explicit. This is not a robot
result, novelty claim, or evidence that StackCube has the same tradeoff.

The robot adaptation uses complete-episode empirical CVaR score weights with a
VaR baseline, not a learned quantile model or a claim of exact recursive episodic
CVaR. The exact score-gradient identity and the clipped PPO surrogate derivative
at the behavior policy passed CPU checks. Finite-batch quantiles, RMS scaling,
per-step PPO clipping and repeated epochs remain approximations. Prior art and
CPU evidence stay in the candidate's research journal.

## Matched continuation, not fresh initialization

All three arms load the **same own-project confirmed reference** actor/critic
checkpoint, whose original seed-1 training used 24,985,600 transitions and 1,088.77
trainer seconds. These prefix costs must be included in total accounting. Its
native SHA256 is
`307a2bc5996d5def9f696ab35b27a9cf96a5cf7401c678ecc981b78170d1084b`;
restricted weights-only conversion exactly reproduces actor SHA256
`39ba198d0c98af14a3d2965ef9a3be859f31aa3c4889db3da311da910c2da153`.
There is no external pretrained policy, teacher dataset or demonstration data.

The optimizer and random stream restart identically for the continuations. This
is explicitly a continuation track, not an uninterrupted-from-scratch comparison.
No prefix policy is re-evaluated and no protected evaluation data enters training.

| Arm | Actor objective |
|---|---|
| risk-ppo-control | Strong ordinary PPO/GAE, including ordinary advantage normalization |
| risk-episode-mean | Full-episode discounted return score (alpha 1), RMS scaled, no minibatch centering |
| risk-episode-cvar10 | Lower 10% full-episode discounted return score with VaR baseline, RMS scaled, no minibatch centering |

Both episode-score arms preserve critic GAE targets and the same actor/critic
architecture. Their matched comparison isolates the risk objective more closely;
ordinary PPO remains the competent practical control. Do not attribute a gain
over ordinary PPO solely to risk if the mean-episode control matches it.

## Execution and protected scoring

- Training seed 1, 1,024 environments, 50-step complete full-horizon episodes,
  gamma .9, entropy .003, LR .0003, 8 epochs, 32 minibatches, original task/reward and
  fixed observation/action/controller interfaces.
- At most 2,048,000 **new** transitions per arm and 300 trainer seconds, including
  startup/built-in evaluation, with 20 seconds reserved for save and an independent
  hard stop. No automatic rerun or fresh lease.
- Sequential training dispatch, max one additional trainer GPU at a time. A
  job's original lease begins only at its own dispatch, not while behind prior
  jobs. Each uses a separate output Volume and durable attempt claim.
- Same reviewed checkpoint bytes/hash supplied to each restricted trainer.
  Trainer receives no benchmark/evaluator/results/credentials.
- Final actor only, scored once on frozen 128-episode development by the existing
  independent evaluator after safe restricted conversion. Max three evaluations;
  no confirmation/private holdout queries or evaluator changes.
- Preserve source/weight hashes, prefix and new costs, all diagnostics and exact
  optimizer steps. The complete new data ceiling is 6,144,000 across the pilot;
  each candidate policy's prefix-plus-new ceiling is 27,033,600 transitions.
- Conservative resource bound $0.97504, including wrapper/startup reserves,
  conversion, all three evaluations and CPU coordination; actual spend and any
  meter lag are reported separately. No extra smoke is included.

The tail mask uses scalar training rewards only. Training final-success flags
are logged solely to measure whether the low-return tail is enriched for actual
failures; changing those labels must not change actor weights (covered by a test).
No ordinary state baseline is subtracted inside a selected tail.

## Decision before results

Continue this version only if the CVaR arm gains at least 3/128 final development
successes over ordinary PPO continuation and does not trail the mean-episode
control. This is an exploratory pilot, not statistical confirmation or a general
method claim. Record all stage and training-tail diagnostics even if the gate
fails. A gain only over the mean-episode control does not beat the strong baseline.
If the mean-episode control wins instead, preserve that distinct observation
without calling it a CVaR benefit. Any replication needs its own outcome-qualified
admission, matched continuation controls and independent prefix training seeds.

Candidate payloads: `research/discovery/risk-pilot-v1/`. The existing 50M quality
batch, its frozen plan and replication readiness remain unchanged.
