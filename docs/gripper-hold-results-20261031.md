# A gripper hold prior did not accelerate early PPO learning

The [single-seed preregistered experiment](gripper-hold-plan-20261031.md) failed its material screen. **Every assessed policy scored 0/128 successful stacks at every checkpoint**: 0, .512M, 1.024M, 2.048M and 4.096M transitions, in all three conditions. Success AUC was zero throughout; T80 and T90 remain censored. The historical 95.9% confirmation result is unchanged.

| Condition, fresh seed 41 | Transitions | Trainer process | Optimizer calls | Learner time | Public scoring |
|---|---:|---:|---:|---:|---:|
| Original strong full 256-unit PPO | 4,096,000 | 250.30 s | 20,281 | 76.63 s | 32,000 transitions |
| Matched 255-unit residual, zero fixed-path output | 4,096,000 | 237.65 s | 20,480 | 65.01 s | 32,000 transitions |
| Matched residual plus analytic gripper hold mean | 4,096,000 | 212.98 s | 20,480 | 54.52 s | 32,000 transitions |

The primary comparison required a +.10 normalized success AUC gain against both controls, with final success preserved. Neither contrast improved at all. This rejects a large improvement within this bounded screen. It does not establish that the prior can never help at a later budget or on another task. All policies remained below the measured success threshold, so the result cannot identify how much longer any condition would need to learn.

Equal transition and process ceilings did not produce equal consumed compute. All runs completed the transition ceiling before their 300-second process cap; the original controller took fewer optimizer steps because its unchanged PPO KL stopping rule triggered. Timing differences include different trajectories and cloud execution variation; they do not establish a reliable implementation speedup.

## What the saved data can explain

The reused independent scorer saved final task outcomes but omitted the adapter's stage diagnostics. This was a reporting mistake: stage rates were planned, but grasp, lift and release labels for these evaluation episodes were not persisted. No extra GPU evaluations were launched to fill the gap after the scoring budget was exhausted. The missing labels do not invalidate success counts, but prevent the planned stage-by-stage causal diagnosis.

The blind training sample retained the first 64/1,024 lanes of all 80 rollouts, without selecting successful episodes. In the first ten rollouts, the hold condition had a larger mean opening (.02086 m versus .01575 m for original PPO). In the last ten, all conditions strongly commanded further closure: mean absolute finger targets were .00106 m, .00071 m and .00019 m for original, masked and hold respectively, despite observed openings around .0223–.0225 m. The hold actor had learned to request closing too. These are descriptive stochastic training statistics, not contact-derived grasp labels or a substitute for independent policy performance. They cannot establish why stacking failed.

Final stochastic training success-at-end was also zero in all conditions. Mean dense reward increased in every run. This agrees with partial reward acquisition without measured successful stacks; it does not identify a particular failure stage.

## Checks, accounting and disposition

The exact original Agent, Gaussian policy, critic, reward, GAE target and PPO update body were preserved. Two masked programs differed only in the fixed hold-path output switch. Runtime task source, dependency versions and PhysX matched pinned hashes. Fixed-path coordinates remained unchanged through training; a deviation would have failed the program. Policies were independently loaded as safe plain MLP artifacts in a separate scorer image; no custom actor code ran inside the scorer.

**26 targeted CPU tests passed** across preparation, analytic mapping, gradient masking, export parity, comparison gates and related pilot analyses. The separate production-Agent CPU stress gate used 64 synthetic optimizer steps and zero simulator interactions. These are distinct from the three real GPU training runs and their GPU scoring.

Total new work was **12,288,000 training plus 96,000 public scoring transitions**, zero source-acquisition transitions, zero demonstrations and zero confirmation/final queries. Both Modal apps are archived and verified stopped with zero tasks. Local training evidence contains 288 files / 231,606,196 bytes; scoring contains nine files / 594,958 bytes, with hashes and exact launch provenance. TensorBoard learner timing was extracted with the pre-existing isolated reader; no environment pins changed. No videos, full critic/optimizer resume states, full 1,024-lane trajectory store or scored stage labels were saved.

The [machine result](../cumulative/results/gripper-hold-20261031.json), [blind action-trace summary](../cumulative/results/gripper-hold-traces-20261031.json) and retained event files preserve the failure. Do not launch an automatic variant or extend these runs. The late-stage proposal-coordinate result from the preceding pilot did not transfer into a large early from-scratch gain under this intervention. Future work needs a new diagnostic question and adequate recorded stage evidence, rather than a longer version of the same screen.
