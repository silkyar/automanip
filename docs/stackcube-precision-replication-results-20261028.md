# StackCube 50M PPO replication: 98.7% mean development success

The constant full-horizon PPO recipe completed all three training seeds at
49,971,200 environment transitions each. Final frozen development success is
98.70% mean, ranging from 98.44% to 99.22%, with a sample standard deviation of
.45 percentage points across seeds. This reproduces high development performance
but does not establish 99% independent confirmation success or 100% reliability.
The replication queried development only. Subsequent once-only
[confirmation](stackcube-precision-confirmation-results-20261028.md) measured
95.90% mean and missed the 99% target. The same 128 development
episodes are paired across seeds; pooling their outcomes does not create 384
independent benchmark scenarios.

| Training seed | 25.6M | 38.4M | Final 49,971,200 | Trainer time | Actual optimizer steps |
|---|---:|---:|---:|---:|---:|
| 1, original quality screen | 117/128 | 126/128 | 127/128 | 38m15s | 110,572 |
| 2, replication | 126/128 | 124/128 | 126/128 | 37m08s | 103,775 |
| 3, replication | 121/128 | 120/128 | 126/128 | 35m09s | 72,352 |

Seeds 2/3 measured subprocess times were 2,228.222694148 and 2,108.697150856
seconds on NVIDIA L40S. Learning curves include the checkpoint's cumulative
trainer time separately from final process duration. KL stopping produces
different actual optimizer counts despite equal environment-transition budgets.
These are additional random initializations, not continuations of seed 1 or
pretrained policies. No earlier checkpoint was substituted for a final policy.

## Remaining failures and interpretation

Every final episode reached the diagnostic grasp, lift, transport, placement,
release and contact-free-placement stages at some point. The five final failures
across the three seeds were labeled `released_not_stable`. Stage attainment is
an episode-level "ever" diagnostic, not proof of a single monotonic action
sequence or a cause of failure.

The separate ten-consecutive-step stability diagnostic was attained in 125/128,
118/128 and 125/128 episodes respectively. It is not the official primary success
criterion and should not be silently substituted for it. The final stock success
count and this additional stability measure answer different questions.

The prior confirmed 25M reference remains 94.08% across three seeds, with its
original independent confirmation results preserved. The new high score used
roughly twice the training transitions. It is evidence for additional training
with the existing recipe, not an equal-budget sample-efficiency discovery.
Late learning is not monotonic: seeds 2/3 dipped at 38.4M before recovering.

## Reproducibility and budget

Recipe/source remain those selected by the predeclared simplicity rule in
`docs/stackcube-precision-plan-20261027.md`. No task, reward, observation/action
interface, horizon, physics or success criterion changed. Protocol SHA256:
`fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764`.
Trainer SHA256:
`cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082`.

Final policy weight digests:

- Seed 1: `ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`
- Seed 2: `c3abf0e70973e39c3616f0a217f75583aeff56aea86ad253370e501a24472959`
- Seed 3: `ec3202403c8cb7c705c1591ce61d61a242d4d9ab957241b4b98d8697bef5cb10`

The replication archive `runs/stackcube-precision-replication-20261027/` contains
119 verified files and 128,760,298 bytes, including all native checkpoints/logs,
safe actor artifacts and six frozen development reports. No native checkpoint
was deserialized locally. Summary SHA256:
`98b363ddc6d61bd8c7bd053b58c70e7d0fd80d8feb3c7c9caae88baaa41e4e6e`.
Combined development summary:
`.runtime/modal-precision-three-seed-development-summary.json`.

App `ap-NhFEBzGjtcIRvSKqTjr0fn` stopped automatically with zero remaining tasks;
no cleanup mutation was needed. Reported replication cost is $2.84351018, below
the $4.50 cap, subject to billing lag. Overall reported session spend after
replication is $10.56756884. The separate $3 BC/AMP envelope remains reserved
inside the same $100 total; no fresh $100 allowance was created.

The user explicitly approved reviewed code/checkpoint/demo uploads and running
this workflow after the earlier approval block. The next small assisted pilot
compares BC-only, standard BC+PPO and BC+PPO with residual-calibrated exploration.
AMP is deferred. The pilot reports teacher training/demo generation
and student compute separately, while preserving the independent scorer and
all earlier results. Independent confirmation was used once as a promotion check;
its reports are excluded from candidate tuning. No private final holdout claim is made.
