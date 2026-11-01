# From-scratch KL/value completion: frozen source handoff

Status: five CPU contract tests pass; **no GPU launch by this agent**. The owner
reviews the two frozen sources before spending. The hypothesis/source were fixed
before the newly reported BC pilot outcomes. No protected scoring episodes,
private seeds, credentials, infrastructure or master ledger were accessed/edited.

## Mechanism and limits

The pinned public PPO computes minibatch policy KL before its joint actor/value
loss. Exceeding the threshold breaks both the minibatch and outer epoch loops,
ending value fitting too. Actor and critic have disjoint parameters. This supports
a structural hypothesis, not a measured value-estimation bottleneck:
**preserve the actor KL stop while letting the original critic objective use the
remaining minibatches within the same configured eight epochs**.

The control retains the original stopping behavior. The candidate freezes actor
mean and logstd at that same first stop; only critic updates run for the suffix.
There are no extra epochs, new models, teacher/demo inputs, checkpoint loading,
target refreshes or reward/interface changes. GAE returns remain fixed. Both
start randomly from the same strong full-horizon PPO constant recipe.

This is a conventional policy/value optimization separation, related to
[Phasic Policy Gradient](https://proceedings.mlr.press/v139/cobbe21a.html).
It is not full PPG: there is no representation-distillation phase, auxiliary actor
head or retained multi-rollout buffer. No novelty or sample-efficiency gain is
claimed. Extra value fitting consumes additional compute and can reinforce biased
targets; if policy KL rarely stops updates, this intervention does little.

## Exact frozen files

Directory: `research/discovery/from-scratch-kl-value-v2/`

| Arm | File | SHA256 |
|---|---|---|
| Strong PPO control | `scratch-kl-stop-control.py` | `de067415f6895f7b4fcc534cff20fe9cd628fef6faae3846c3b49687ff526632` |
| Critic completion | `scratch-kl-stop-critic.py` | `8784f4d7aeb268a4794a59c509b761f740fffdd5932b0e6dd364ec4157c9228b` |

Reviewed public base SHA256:
`cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082`.
Builder `research/discovery/critic_kl_completion_v2.py` SHA256:
`b817a73a8c39d506ab8cf37970b64685488986e6e5a0ed62738f2b160568768a`.
`source-receipt.json` records these identities and CPU contracts. Removing
`KL_VALUE` blocks restores the exact public base. Agent/serialization and original
actor update, ratio, entropy, reward and value-target code remain intact.

Use these **v2 payloads**, not the earlier v1 one-extra-epoch kernel used for the
small regression counterexample. No frozen BC/AMP/residual-std payload was edited.

## Actor and RNG isolation

The shared Adam optimizer clears gradients with `set_to_none=True` before every
critic-only update. Actor gradients must remain absent; its parameter values,
Adam moments and step counters therefore do not advance. A CPU counterexample
shows that zero-valued actor gradient tensors would allow stale Adam momentum to
move the actor. Critic-only clipping uses the existing .5 limit on active critic
gradients. A deadline skip also clears stale gradients and makes no optimizer call.

Extra epoch shuffles would otherwise change future actor minibatch randomness.
Both sources capture NumPy RNG state at the first KL stop; after the candidate's
critic-only suffix it restores that state. The next rollout therefore starts
from the control's shuffle RNG position. No dropout or random critic operation
requires additional Torch draws.

Five CPU tests execute the **actual embedded PPO update block**. A forced stop
produces one joint/actor update in both arms. Candidate then makes 15 critic-only
updates, for 16 critic updates across eight configured epochs. Its actor/logstd,
actor Adam moments/step counters, action log-probabilities, entropy and NumPy/Torch
RNG states match control exactly. Critic fixed-target fit improves in this test.
With KL stopping disabled, both arms make 16 joint updates and all parameters match
bitwise. Other tests cover stale-momentum drift, deadline skips, source stripping,
unchanged Agent schema and fixed value targets. Test runtime was .813 CPU seconds;
these are implementation checks, not robot-learning evidence.

The earlier bounded regression probe is also preserved in
`research/discovery/critic-completion-cpu-gate-v1.json`: extra fitting improved
clean-target held-out MSE in 5/6 paired toy seeds (mean .18810→.18254), but two
deliberately biased-target cases worsened true rare-phase error despite improving
training-target fit. In one case, true error rose 2.912→3.759 while training-target
MSE fell .22335→.17564. This rejects “more value fitting always helps”; it does not
measure StackCube bootstrap bias or predict a policy gain.

## Exact counter receipt

One `AUTOMANIP_KL_VALUE_UPDATES` JSON object is emitted per rollout. Its exact schema
is `update-receipt.schema.json` beside the frozen payloads. Numeric fields are
also logged under `audit/kl_value/`.

Count definitions, per rollout and cumulatively:

```text
actor_optimizer_calls = joint_actor_critic_optimizer_calls
critic_optimizer_calls = joint_actor_critic_optimizer_calls + critic_only_optimizer_calls
optimizer_calls_total = joint_actor_critic_optimizer_calls + critic_only_optimizer_calls
```

These are measured optimizer-call counts, not assertions that every tensor value
changes. Existing `audit/optimizer_steps` keeps its original **joint/actor-call**
meaning. Full accounting must use the new cumulative totals, not that old tag
alone. Row counts are minibatch presentations of existing rollout data, not new
independent transitions.

The receipt includes source identities, cumulative environment transitions,
configured/reached epochs, first KL-stop epoch/minibatch (zero-based), observed KL,
critic-only rows/time, full optimizer-loop time, deadline skips and value loss.
It reports zero actor updates after KL stop, zero target refreshes and no new
models/teacher data. The original trainer/launcher must also retain complete
simulation, initialization, training, save and evaluation wall/cost accounting.

Cross-checks for the owner:

- Control always has zero critic-only calls. With no KL stop, both methods follow
  identical joint updates.
- For a candidate stop with no deadline skips, joint plus critic-only calls equal
  `8*num_minibatches`. With skips, add skipped minibatches to that identity.
- First stop position and joint behavior agree between methods until their critic
  states diverge on subsequent rollouts; the current actor stays frozen thereafter.
- At a stop, actor/logstd and their optimizer state stay fixed throughout the
  suffix. NumPy state is restored before the next policy shuffle sequence.
- Count all additional critic calls, rows and time; they are not free data reuse.

## Owner-reviewed pilot proposal

Two matched **from-scratch**, seed 1 runs, no checkpoints or demonstrations:
10,240,000 requested transitions and 600-second ceiling per arm, on the same L40S
software/hardware recipe. The owner proposes a combined $3 cap including training,
evaluation and overhead, with independent hard stops and save reserves. No launch
is authorized by this document.

Use the existing fixed precision/CAPG pipeline and preselected development points
at 5,120,000 and final 10,240,000 transitions, at most four development calls total.
If a wall cap prevents a point, report it as unreached/censored; do not extrapolate
an equal-step result. The trusted scorer, task, horizon 50, state48/action8,
controller, reward and artifact validation remain unchanged.

Common CLI settings are gamma .9, GAE lambda .9, entropy .003, LR .0003, eight
epochs, 32 minibatches, 1024 environments, 50 steps, full-horizon resets and target
KL .1. The only scientific difference is the critic-only suffix. Existing
checkpoints/evaluation disabling/safe exports remain owned by the trusted launcher.

Prediction: better value fitting on already collected experience improves the
transition-indexed learning curve, possibly at lower throughput. Falsification:
no learning-curve improvement, poorer final quality, negligible triggering, or only
training-target fit gains without policy improvement. One paired seed is a screen;
the owner retains promotion/confirmation rules. Current source should remain
frozen during review and any pilot.

## Separate residual-std production review

The frozen BC+PPO residual-std source was inspected again; no production blocker
was found. Its new, separate exact initialization-receipt schema is
`research/discovery/bc-residual-exploration-v2/calibration-receipt.schema.json`.
Infrastructure received identity, split-disjointness, row-count, clamp, artifact,
timing and negative-control requirements. It must not import candidate code or
decompress candidate NPZ in the trusted evaluator. Any independent RMS recompute
belongs in a separately bounded training-side audit.

The owner subsequently reported BC-only 49/128, ordinary BC+PPO 0/128 and residual-std
BC+PPO 0/128 after the same 5.12M new transitions. This is a negative development
screen, not evidence identifying the cause of lost BC performance. Do not extend
or tune that version to rescue it. The from-scratch KL/value hypothesis above was
fixed before those outcomes and uses none of their episodes or teacher data.
