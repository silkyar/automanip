# Which entropy route caused the recovery of early grasping?

Registered before new training. This is one causal question with two narrowly
specified route interventions, not a collection of new algorithms. Lowering
initial alpha from 1 to .005 changed both the actor objective and soft critic
targets; at 102,400 transitions the same seed's grasp count changed from 0/128
to 64/128, but lift only from 0 to 1 and stack stayed 0. Neither diagonal passed
the original combined grasp/lift gate. All historical results remain unchanged.

The later physical probe failed its matching gate before intervention. Current
surrogate gradients do not establish physical returns or explain the early
temperature transient. Another replay-prefix attempt or generic saturation
remedy is therefore not the next experiment.

## Two nested route tests

| New arm | Actor/temperature path | Soft target entropy coefficient |
|---|---|---|
| high-actor-low-backup | Original alpha initialization 1, original adaptive update | .005 times that arm's alpha |
| low-actor-high-backup | Original low-alpha initialization .005, original adaptive update | 200 times that arm's alpha |

Reuse the cached high/high and low/low 102.4k reports; do not retrain or rescore
them. Both new arms start from the identical seed-1 random actor, critics and
target critics. No checkpoint, demonstration, diagnostic data, analytic controller
or simulator query enters training. Observation/action/reward/task/evaluator,
64 environments, learning starts, replay, minibatch size, update exposure,
optimizer, gamma .99 and network architecture are the unchanged fast recipe.

Implement only a multiplier in the soft target entropy term. Relative to the
corresponding diagonal learner, actor and temperature update code are identical.
The low-initial-alpha arm sets the existing scalar parameter before any optimizer
state exists, using the exact production float32 log(.005). The same optimizer
retains its reference to that parameter.

For a fixed batch, next action/logprob, mask and current target critics:
`y(mult)-y(1) = gamma*(1-stop)*alpha*(1-mult)*next_logprob`.
This is not constant reward/value centering. Its scale is state/action dependent.
Check this identity, termination masks (including timeout bootstrapping), unchanged
current Q outputs and actor/temperature losses/gradients, paired initial tensors
and random-generator position on CPU before admission. Independent source review
must pass. No surrogate probe constitutes a robot result.

Adaptive temperature and the policy subsequently feed back into each other. The
absolute target doses can diverge across arms. This is a pair of closed-loop
route-sufficiency interventions, **not a fixed-alpha factorial or unbiased
factorial main-effect estimate**. Log actor alpha and effective backup alpha.

## Outcomes fixed before launch

One fresh seed-1 run per crossed arm, exactly 102,400 transitions maximum, 150
child-process seconds maximum each. Primary is a single frozen development score
at exactly 102,400 transitions per unique policy hash (reuse any durable report).
Keep the existing 51.2k safe export/logs without an additional protected query.
Censoring or missing primary is missing evidence, not permission to extend.

For this **grasp-route** question only, each crossed arm with grasp >=64/128 is
recovery-supporting, <=8/128 weakens that route's sufficiency, and intermediate
counts are inconclusive. Always report lift, transport, placement, release,
stability and stack too. If only high-actor/low-backup recovers, lower target
entropy is sufficient in this closed loop; if only low-actor/high-backup recovers,
the actor/temperature path is sufficient. If neither recovers, neither isolated
route suffices under the declared doses. If both recover, both are sufficient
under their respective interventions; there is no unique-mediator conclusion.
These are single-training-seed, adaptive development observations, not population
causal effects or demonstrations of lifting, stacking, 10x efficiency, or a strong
new baseline. No confirmation, automatic replication, promotion or extension.

## Admission and accounting

At most two L40S workers, each child <=150s, function <=210s plus30s startup;
one coordinator <=900 CPU seconds; at most two frozen development functions
<=120s plus30s startup. One shared $2 reservation under the existing total $100
Modal cap. The user explicitly resumed bounded Modal experiments on Oct29;
Codex stays local and no Runpod spend is authorized. A lease begins at the
single submitted worker call and is never refreshed. Durable claims prohibit
duplicate campaign/worker execution. Archive every attempted arm and evaluator
report, log queue/image time separately, verify STOPPED/zero tasks, and close
the reservation only after verified archival. Failed/censored arms are retained.
