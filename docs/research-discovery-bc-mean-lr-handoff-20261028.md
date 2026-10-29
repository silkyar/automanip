# Assisted BC residual-std mean-LR proposals — 2026-10-28

Status: two self-contained **local proposals only**, not launched. Existing BC,
residual-std v2, AMP, and from-scratch critic-completion sources remain frozen.
This is a teacher-assisted track, with teacher training, collection, BC training,
and student compute counted separately and in total. No teacher weights enter
the student. No novelty or robot-performance improvement is claimed.

## Hypothesis and evidence limit

The completed public TRAINING audit reports exactly one joint optimizer step per
rollout in residual-std BC+PPO (100 total; logged approximate KL .21824–6.01261,
target .1), versus 22,782 in original-std BC+PPO (KL .01999–.13188). Both final
development aggregates were 0/128 after 5.12M new steps; BC-only was 49/128.
These aggregate findings motivate a noise/step-size interaction hypothesis. They
do not establish the cause of either failure, and no protected episode data are
used here.

The separate [Gaussian diagnostic](research-discovery-gaussian-kl-20261028.md)
uses public CPU demo states, own Gaussian samples, and synthetic advantages.
It shows that smaller sigma can sharply amplify mean-step KL while fresh Adam
mean parameter movement stays similar. Its default-sigma synthetic update also
overshot the KL threshold. Neither that diagnostic nor the present code proves
that lower LR improves robotics learning; lower LR can simply suppress learning.
Gaussian trust-region geometry and adaptive-gradient scaling are established
ideas: [TRPO](https://proceedings.mlr.press/v37/schulman15.html),
[PPO](https://arxiv.org/abs/1707.06347), and
[Adam](https://arxiv.org/abs/1412.6980).

## Exact intervention

Both payloads start from the exact frozen residual-std v2 source
`c5ca37a6ff22ddd2e89288f5697e3041507e834adf8f8731a49a17d1423fd200`.
Its existing production loader and BC training/validation split are unchanged.
It calibrates eight residual stds from TRAIN episodes, excludes validation
episodes, clamps to [.05, exp(-.5)], and initializes existing trainable logstd.

The new code derives a fixed initial ratio
`r = exp(mean(log(initial_sigma))) / exp(-.5)` from those initialized parameters
in float64. No CPU-probe constants, validation states, teacher policy parameters,
or task scores enter the ratio. Expanding the existing logstd in float64 can
differ by final rounding from the existing float32 std audit; comparison should
allow ordinary float32 rounding, not different calibration data. The ratio is
fixed at initialization even as PPO subsequently learns logstd.

After the original fresh Adam constructor, before any optimizer state exists:

- Group 0 retains critic plus actor_logstd, original LR and all Adam defaults.
- Group 1 contains only actor_mean, with LR `original_active_lr * r` or
  `original_active_lr * r**2`.
- Epsilon remains 1e-5. The original joint loss, global gradient clipping,
  entropy, KL-stop logic, optimizer calls, task reward, actor/critic architecture,
  observation/action interface, serialization, and scheduler values are unchanged.

Each original constant-scheduler reset is immediately followed by multiplier
reapplication. The optional original annealing reset is also immediately
followed by reapplication. It never compounds the multiplier. Existing
`charts/learning_rate` now reads the unscaled critic/logstd LR in group 0;
`audit/assist_mean_lr/mean_lr` records the actual mean LR. LR audit occurs after
all scheduler resets and before that rollout's optimization. All minibatch
optimizer steps in that rollout use these recorded group LRs.

No extra BC forward pass is performed. The existing residual calibration report
is retained in memory to bind the new receipt to its exact data and actor.
Group-initialization overhead is logged and belongs inside the full job lease.

## Frozen source handoff

| File under `research/discovery/` | SHA256 |
|---|---|
| `bc_mean_lr_v3.py` | `a7f7f7381ddd7b32f0bd19bbbcfa4d20f11d966ec9231aa345a70e7e9b28e147` |
| `bc-mean-lr-v3/assist-bc-ppo-residual-std-mean-lr-r1.py` | `bd471bbe21528ba42d2e29dadfef8ff14697150c92129e01038201edc6bef55d` |
| `bc-mean-lr-v3/assist-bc-ppo-residual-std-mean-lr-r2.py` | `ec661250f7fb974cb0e98780fe3b03325505de05ba00a159b614d11071a18238` |

`source-receipt.json` and strict `lr-receipt.schema.json` accompany the payloads.
They import no local candidate or trusted evaluator modules. Removing every
`BC_MEAN_LR` marked block recovers the exact residual v2 bytes; subsequently
stripping its residual blocks recovers the original assisted v1 base.
The existing `Agent` AST and state_dict schema match the base exactly.

CLI contract: replace only the script path in the previously reviewed residual
BC+PPO command. Preserve every other argument, seed, version guard, episode
horizon, lease, observation/action contract, and output conversion. No new CLI
options or environment variables are introduced. Required existing inputs are
`--assist-dataset`, `--assist-bc-actor`, and `--assist-bc-report`; production data
and student BC actor must have the identical digests used by the residual
comparator. Neither source accepts local CPU smoke data in production.

## Receipt checks for a future reviewed pilot

`AUTOMANIP_EXPLORATION_INITIALIZATION` remains intact. The new
`AUTOMANIP_MEAN_LR_INITIALIZATION` record adds the ratio/power/multiplier, initial
mean/critic/logstd LRs, eight initialized stds plus canonical-JSON digest,
runtime source and patch/base hashes, data/BC actor digests, training and excluded
validation episode indices, original residual RMS/old std, existing calibration
rows/time, group-initialization time, and zero extra BC rows/optimizer/env steps.

`AUTOMANIP_MEAN_LR_ACTIVE` appears once per optimized rollout, with iteration,
global step, actual mean/critic/logstd LRs and fixed multiplier. Scalar tags are
`audit/assist_mean_lr/{ratio,mean_lr_multiplier,initial_base_lr,initial_mean_lr,initial_critic_lr,initial_logstd_lr,additional_bc_forward_rows,group_initialization_wall_seconds}`
at initialization and `{mean_lr,critic_lr,logstd_lr,mean_lr_multiplier}` per rollout.
The existing eight `audit/logstd_dim_*` values continue to report learned std
parameters; existing eight initial std tags remain present.

Training-side/receipt review should verify:

1. Source hashes, dataset/BC actor/report identities, exact same train and excluded
   validation lists as the original initialization, and disjoint splits.
2. Eight stds within clamp tolerance; ratio and power reproduce the multiplier;
   active mean LR equals base LR times that fixed multiplier. Critic and logstd
   LR match the original scheduler. The initial std digest uses JSON with
   separators `(',', ':')` and list order preserved.
3. Residual calibration rows/time are the existing pass; additional forward rows
   are zero. Include teacher, generation, BC, calibration, initialization, and
   full student time/counts in the assisted-track totals. Count every optimizer
   call even if a new candidate still stops after one minibatch.
4. Final artifact retains the ordinary actor/critic schema; optimizer groups are
   training-only. Do not execute candidate Python or decompress candidate data
   inside the trusted scorer for these checks.

## CPU gates and negative controls

Four focused CPU tests passed in .669 s (zero simulator transitions, no GPU):

1. At r=1, regrouped Adam and the original single group produce bitwise identical
   parameters and per-parameter Adam moments/steps across three joint updates
   with the original global clipping rule. Regrouping consumes no Torch RNG.
2. With the same fixed gradients, only mean displacement scales; critic/logstd
   parameters and Adam state match exactly. Epsilon and defaults remain intact.
   Both original scheduler resets preserve the initial multiplier.
3. Both generated payloads strip bytewise to the base, preserve `Agent`, execute
   their own embedded helper functions, and emit correctly bound receipts.
4. Already-used optimizers are rejected, preventing regrouping initialized Adam
   state silently.

The earlier synthetic mean-KL diagnostic provides the control where an aggressive
LR reduction suppresses KL. It supplies no task-learning target and is not a
reason to tune either power further. The existing lower-noise exploration
counterexample remains relevant: a low-variance cloned policy can fail to find
needed corrections even when it follows demo actions closely.

For a future owner-reviewed comparison, hold residual std, BC actor/dataset,
critic schedule, reward and all other training arguments fixed. Compare the
original residual control against these two predeclared powers at the same new
transition/time caps and selected learning checkpoints. Reusing the historical
control requires identical production artifacts and disclosure that it was not
concurrent. KL compliance or increased update counts alone is insufficient:
reject a variant that only restores counts without better learning, or that
slows useful correction learning. Independent full training seeds and protected
verification remain owner responsibilities if a pilot is promising.

The from-scratch critic-completion pair retains priority. These assisted
proposals are separate from critic-completion changes and do not repair or extend
the recorded failed BC pilot.

```sh
.venv311/bin/python -m unittest discover -s tests -p 'test_discovery_bc_mean_lr_v3.py' -v
```
