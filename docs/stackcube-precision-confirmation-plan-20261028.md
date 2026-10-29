# Once-only confirmation of the fixed 50M PPO recipe

After the two-seed replication, final development counts are 127/128, 126/128
and 126/128. This plan is registered before viewing confirmation outcomes.
All three final policies at 49,971,200 training transitions are fixed; no earlier
checkpoint or alternative schedule can replace one after a result is seen.

Evaluate exactly 512 frozen confirmation episodes per policy, three training
seeds total. No new training, tuning, demonstration generation or development
query is part of this campaign. The target is mean success at least 99% and
every training seed at least 98%. Failure of that target is recorded as such,
not repaired by another confirmation query. Estimates and individual counts
are reported regardless of the binary target.

Use the unchanged frozen evaluator function, benchmark distribution, task,
policy validator, dependency pins, horizon and success definition. Bind the
three exact safe-policy digests in `deploy/modal/precision-confirmation-plan.json`.
Only this function instance's static digest-admission file is supplied separately;
the earlier promotion allowlist and evaluator source remain unchanged. Root
reviews admission, contract and driver; the autonomous researcher cannot edit
them or access the protected reports. Existing GPU attempt keys prevent repeats
across apps, restarts and transport failures. No key is deleted or refreshed.

This is the existing independent confirmation distribution previously used for
the older 25M recipe. It is not a new private final holdout, nor a set to query
after every research experiment. It measures a fixed promoted recipe once;
candidate training and subsequent pilot selection must not consume its reports.

At most three L40S scorers, 240-second GPU-container limit and 220-second evaluator
subprocess limit, no retries. One small trusted CPU coordinator has an 1,800-second
ceiling and persists claims, submission IDs, full reports and failures. The
conservative resource bound is $0.5151807; reserve **$1 within the same $100
session cap** for setup/egress/meter lag. No unrelated resources are changed.
All outputs are archived and verified, exact app stopped/zero-task status checked,
and provider-reported spend reconciled before releasing the reservation.

Three CPU admission/parity tests passed: disabled admission is rejected; policy swaps,
split/episode/concurrency/time/budget changes, retries, training and acceptance-rule
changes are rejected. Extracted scorer bytecode/constants match the original frozen
function, and registration passed against the pinned SDK locally. No new GPU test is claimed before submission. This driver
reuses the already exercised frozen scorer and numeric-only policy transport.

The first deployment failed in the local entrypoint before any coordinator/GPU
call: SDK 1.6.2 does not permit image replacement via `with_options`. That failed
setup is preserved and billed within session accounting. The corrected adapter
registers the exact hash-bound trusted function body against the separate
admission image, with no scorer-body changes. The same campaign and GPU once-keys
are retained; no evaluation claim was deleted or refreshed.
