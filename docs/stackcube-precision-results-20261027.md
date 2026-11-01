# StackCube quality screen: 99.2% development success in one seed

Continuing the strong full-horizon PPO recipe to 49,971,200 transitions reached
127/128 development successes (99.21875%). This is a single training seed,
selected using development episodes. It has not established 99% across training
seeds or independent confirmation episodes. The earlier confirmed three-seed
25M reference remains 94.08%; its checkpoints and results are preserved.

## Matched results

All three arms used random initialization with seed 1, the stock StackCube task
and reward, gamma .9, initial entropy coefficient .003, learning rate .0003,
1,024 environments, 50-step full-horizon rollouts, eight PPO update epochs and
32 minibatches. The two schedules changed only after 25.6M transitions.

| Method | 25.6M success | 38.4M success | 49,971,200 success | Trainer time | Actual optimizer steps |
|---|---:|---:|---:|---:|---:|
| Constant settings | 117/128 | 126/128 | 127/128 | 2,295.408 s (38m15s) | 110,572 |
| Entropy annealed to zero | 117/128 | 116/128 | 121/128 | 2,172.979 s | 84,080 |
| Learning rate annealed to .00003 | 117/128 | 123/128 | 127/128 | 2,466.694 s | 156,882 |

The three actors at 25.6M were byte-identical. Their single evaluation was reused:
nine curve entries required seven distinct frozen 128-episode evaluations, not
nine independent queries. KL early stopping explains why a fixed transition
budget does not imply equal numbers of optimizer steps; the counts are measured.

The preregistered simplicity rule selects constant settings. Learning-rate decay
ties its final count, while entropy decay performs worse. The evidence supports
more training with the existing recipe, not an improvement in sample efficiency.
The selected run used roughly twice the transitions of the confirmed reference.

## Artifacts and next decision

The complete verified archive is
`runs/stackcube-precision-screen-20261027/` (176 distinct files, 192,969,983 bytes).
The selected safe actor is
`precision-constant/step-49971200/`, weights SHA256
`ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`.
The terminal summary SHA256 is
`21e300cf4467ee9dbeca1b91d35a3b5f8d6358f532bdf443f0ae939be95d2d2f`.
The unchanged frozen protocol SHA256 is
`fe2cc5987576a84b7a66ae9c8c9c4099b688c682d5736c5388217f5372391764`.

Seeds 2/3 replication is prepared with a $4.50 cap, but no replication was
submitted. Automatic approval review rejected the new private upload scope;
standing approval for the night's reviewed Modal workflow is pending. All
completed apps are stopped. This screen's reported provider cost was $4.48262164,
subject to billing lag; the shared ledger accounts for the overall $100 cap.

After replication, independent confirmation would be a promotion decision rather
than a feedback source for tuning. No private final holdout result is claimed.
The user's next priority is lower training time/sample use with performance
retained. Demonstration-assisted BC/PPO/AMP will be tracked separately and must
include the teacher's 49,971,200-transition, 2,295.408-second training prefix,
demonstration generation, BC updates and student RL costs.

Operational details and source receipts:
`docs/modal-precision-screen-20261027.md`. Predeclared decision rule:
`docs/stackcube-precision-plan-20261027.md`.
