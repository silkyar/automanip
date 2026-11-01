# Early grasping regression: discount was insufficient; initial alpha partly matters

2026-10-28. Two prespecified one-factor tests followed the negative replay-moment
screen. Same frozen task/reward/evaluator, paired random actor initialization,
one training seed and exact common development checkpoints. Neither is a new
algorithm, a multi-seed claim or a sample-efficiency improvement in stacking.

## Main result

Counts below are grasp/lift/final-stack out of128 independent development
episodes per policy. The same development distribution is reused across methods.

| Fast-recipe intervention |51,200 transitions|102,400 primary|256,000 context|Trainer process seconds|
|---|---|---|---|---:|
| Cached gamma.99 / initialalpha1 control |0/0/0|0/0/0|0/0/0|586.449 to874,880 total|
| Only gamma.99→.9 |0/0/0|0/0/0|4/0/0|151.354 to256,000|
| Only initialalpha1→.005, gamma.99 |49/0/0|64/1/0|121/0/0|195.434 to256,000|

The [discount gate](replay-discount-sufficiency-plan-20261028.md) weakens gamma
restoration alone being sufficient. The [temperature gate](replay-temperature-sufficiency-plan-20261028.md)
is **partial/inconclusive**: primary grasp reaches64, but lift1 is below32.
The later121/128 grasp result does not replace the primary point or pass the
combined grasp/lift criterion. No candidate stacked; no confirmation or automatic
replication was launched. Historical95.9% StackCube confirmation remains a
different50M PPO result, not this95% grasp metric.

The temperature intervention is nevertheless a useful seed-local causal signal:
with the same102k data allowance and6,088 critic/2,029 actor calls, changing only
initial alpha recovered grasping. It did not recover transport. A reduced
entropy incentive and smaller initial soft-value targets are both mediators;
this experiment does not distinguish them. It does not establish that stronger
entropy is generally harmful or that a low-alpha algorithm will learn stacks.

## Why these factors were selected

The earlier adapted SAC had grasp128/lift115 at102k and grasp128/lift121 at256k,
but zero stacks. At102k it took515s, used49,216 actor/critic calls and alpha.002519;
the fast control took71s, used6,088 critic/2,029 actor calls and alpha.2336.
Many other settings also changed: critic depth, batch size, vector count, EMA,
discount, learning rate/betas, warmup and temperature sampling timing. That old
comparison was observational and **not matched compute**. The tests above isolate
one setting each within the fast recipe, rather than attributing the historical
gap wholesale to a favorite mechanism.

A fixed-policy scalar check showed that UTD/EMA/discount together can alter the
target-backup relaxation clock160× in transition units; gamma-only restoration
still leaves16×. This is an idealization, not a neural target reconstruction.
The negative gamma result narrows its sufficiency interpretation.

A separate fixed-entropy scalar Adam check reproduced the fast temperature
clock:2,029 updates take alpha1 to.234076, whereas an initial.005 gives.001170.
Actual robot entropy and feedback remain learned, so these numbers are not a
robot rollout prediction. The one-line alpha intervention was selected before
its outcomes, using the order of the historical51k temperature, not confirmation.

## Numerical values are not enough

Gamma restoration substantially reduced Q/target scale without restoring early
grasping. At102k its last logged Q loss was.0142 and target mean16.07, versus the
cached control's loss.2375 and target mean64.47; both scored0 grasp/lift. These
are last-minibatch statistics, not true-value error or a direct mechanism proof.

The independent [constant-offset CPU experiment](value-offset-probe-20261028.md)
also found that value RMSE can increase to15.7–15.9 while action-gradient RMSE
changes only18–32%. Its predefined gate is partial/inconclusive. A fixed value
coordinate shift did not improve it; oracle initialization is informative
assistance, not an automatic centering method. These findings prevent equating
stable/accurate-looking values or lower TD loss with correct policy improvement.

## Next question and preserved evidence

The next diagnostic examines the aligned64-episode/3,200-transition saved
temperature-run window at256k. It distinguishes short-lived grasp flags from
sustained grasp without upward motion, using actual successors/actions and
verified observation coordinates. It is training behavior with parameters
changing during collection, not a frozen-policy or additional benchmark sample.
No new simulation or fitting is required for that inspection. The identical
final/256k snapshot is counted once.

Archives `runs/gamma-sufficiency-20261028/` and
`runs/temperature-sufficiency-20261028/` retain every verified source/input,
receipt, numeric diagnostic state, safe policy and3 trusted development reports.
Initial safe actor identity was independently checked before scoring. Shared
core source was unchanged; the alpha payload differs in exactly one learner
line and its canonical import mapping is explicit. Independent source reviews
completed; one model-capacity failure was resolved by a bounded review retry.
Neither task was resubmitted or rescored.

Additional training:512,000 transitions total,2 fresh seed1 learners,6 once-only
development queries,0 confirmation queries,0 demonstration/teacher/model/planner
queries. Both apps are archived and stopped with zero tasks. Shared-session
reported Modal cost after closeout:$13.90920166; metering can lag. No new Runpod
spend. All historical PPO/PickCube runs remain unchanged.
