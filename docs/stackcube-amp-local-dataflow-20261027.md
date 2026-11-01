# Local demonstration-assisted pipeline check

This check exercised teacher collection, behavior cloning from random student
weights, safe artifact export and fresh student rollouts. It used local CPU
PhysX with ManiSkill 3.0.0b21, SAPIEN 3.0.4.dev20261009+7dba23f1, Torch 2.6.0
and NumPy 1.26.4. Its simulator differs from the frozen Linux/CUDA reference:
**these outcomes are not benchmark comparisons or evidence of GPU sample
efficiency**. Production loaders reject these CPU demonstrations and BC reports.
No data was uploaded, no cloud job launched, and incremental cloud cost was $0.

## Collection and behavior cloning

The teacher is our own 49,971,200-transition PPO policy, safe weights digest
`ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`.
Its original training took 2,295.408 seconds and must be included in total-cost
reporting for this assisted track; it is not a free pretrained initialization.

Fresh public training seeds 90,173,100–90,173,355 generated 256 complete
50-step episodes (12,800 transitions) in 14.008 local CPU seconds. All 247
successful and nine failed episodes remain in the raw dataset. The BC learner
explicitly selected successful episodes and split by whole episode: 223 for
training, 24 for action-validation. Neither protected evaluation episodes nor
their trajectories were used. Dataset digest:
`36c514c51f26d0cf034cdb49589226d077454d0cb4380571fad05dc4b6b9d04e`.

A random student, training seed 1, learned only from state/action examples;
teacher weights were not a learner input. It completed 5,000 Adam updates
(5,120,000 sampled demonstration rows) in 11.768 local CPU seconds. Initial
training-probe action MSE was .378924; final held-out demonstration action MSE
was .0101664. These are different subsets, so this is not a paired validation
improvement measurement. BC consumed zero new student environment transitions.
The final student weights digest is
`c13416787d05347ccc1e92ad95d48a1bb649b6d7a83e883823600db253b15a53`.

## Export and rollout dataflow

The trusted JSON loader independently accepted the random student, BC student
and teacher artifacts. On 512 actual demonstration observations, a plain Torch
MLP reconstructed from the validated tensors agreed with trusted NumPy inference
to less than 9.7e-7 maximum absolute action error for all three artifacts. No
candidate policy implementation was imported into that rollout probe.

On 64 paired fresh public training seeds 90,174,000–90,174,063, each run for the
unchanged 50-step horizon:

| Artifact | Final task successes | Diagnostic transitions |
|---|---:|---:|
| Random student | 0/64 | 3,200 |
| BC student | 21/64 | 3,200 |
| Teacher | 61/64 | 3,200 |

This supports using the student as a meaningful pipeline input, while showing
that low action MSE alone does not establish reliable stacking. Compounding
state-distribution errors and action ambiguity are hypotheses to investigate,
not conclusions from these counts. This probe did not use the frozen evaluator.

A separate 64-episode-per-arm collection at seeds 90,175,000–90,175,063 retained
actual random-student and BC-student state→successor pairs for a discriminator
component check, adding 6,400 transitions. Raw failures remain present. Success
labels are not discriminator inputs. This collection is also local CPU dataflow
only, not another benchmark query. Across these three local collections there
were 28,800 transitions, plus a preceding 50-transition teacher preflight.

## Prepared scientific comparison

The next paid pilot requires fresh demonstrations on the pinned Linux/CUDA
stack. Compare BC-only, BC+PPO and BC+PPO+AMP from one shared random-student BC
artifact, with bounded student training and the unchanged independent scorer.
Report teacher training, demo generation, BC and discriminator work separately
from student RL transitions and wall time. The full state48→state48
discriminator is a task-specific manipulation prior; its ability to distinguish
trajectories does not by itself prove that it learned useful temporal structure.

The subsequent bounded discriminator probe used 223 teacher training episodes
and 52 BC training episodes, with separate whole-episode diagnostics. After
200 updates (102,400 expert/learner minibatch rows; .904 CPU seconds), held-out
teacher style averaged .82555 versus .63298 for BC. Balanced sign accuracy was
.6875. Shifting teacher successors by 25 steps within each episode reduced style
only .01811, and unseen random-policy transitions scored .71188 despite zero
task successes. These controls limit the interpretation: the classifier signal
is present, but substantial temporal modeling and useful policy improvement
are unproven. In particular, this style score is not a success predictor.
Only a small matched pilot is warranted, with separate task/style reward logs;
the discriminator result does not justify a larger campaign. Full diagnostic:
`research/discovery/amp-local-discriminator-probe-v1.json`. The probe added no
simulator transitions or cloud cost.

The research payloads are prepared and nine focused CPU unit tests pass,
including true terminal successors, reward bounds/detachment, whole-episode
splitting, numerical export, unchanged PPO artifact schema and self-contained
runtime initialization. Four replication and five risk-adapter contract tests
also pass. No new GPU test of the AMP pipeline has been run. Paid launches and
new uploads remain blocked by automatic approval review pending the standing
user approval question; there is no running Modal workload.

Reproducibility files:

- `research/discovery/local-cpu-demos-smoke-v1/{dataset.json,trajectories.npz}`
- `research/discovery/local-cpu-bc-smoke-v1/bc-report.json`
- `.runtime/cpu-bc-rollout-probe.py` and `.runtime/cpu-bc-rollout-probe.json`
- `research/discovery/local-cpu-learner-pairs-smoke-v1/collection.json`
- `research/discovery/amp-prior-assisted-v1/source-receipt.json`
