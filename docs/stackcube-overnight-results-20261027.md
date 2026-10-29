# StackCube overnight results — 2026-10-27

Best development mean: **83.59375% (321/384)** with discount 0.9 and entropy coefficient 0.003. The three training seeds achieved 89.0625%, 75.0%, and 86.71875%. This improves the original official baseline development mean of 31.5104% by 52.0833 percentage points. The 90% mean target has not yet been achieved.

All results use the same frozen independent development evaluator: 128 episodes per training seed. No confirmation or private holdout episodes were queried. These are adaptively selected development results, not a confirmed promotion.

## Full-budget results

Every run requested 25M transitions and completed 24,985,600, from random initialization. Task, reward, observation/action interfaces, policy architecture and evaluator are unchanged. Training time below includes the official trainer process; independent evaluation adds time.

| Recipe | Seed 1 | Seed 2 | Seed 3 | Mean | Mean trainer minutes |
|---|---:|---:|---:|---:|---:|
| Official discount 0.8 baseline | 23.4% | 47.7% | 23.4% | 31.5% | 23.6 |
| gamma090 | 56.2% | 78.1% | 78.9% | 71.1% | 22.2 |
| gamma095 | 7.8% | 71.9% | 25.0% | 34.9% | 21.0 |
| tune-gamma0875-control | 63.3% | 73.4% | 75.0% | 70.6% | 21.8 |
| tune-gamma0925-control | 89.1% | 82.0% | 72.7% | 81.2% | 21.3 |
| tune-entropy001-control | 43.0% | 68.0% | 84.4% | 65.1% | 21.9 |
| tune-entropy003-control | 89.1% | 75.0% | 86.7% | 83.6% | 22.4 |

Discount 0.925 gained on average but lost 6.25 percentage points on seed 3 versus discount 0.9, exceeding the preregistered five-point regression tolerance for anchor selection. Entropy 0.003 met the selection rule: mean gain 12.5 points, worst seed change −3.125 points. Selection here means a research anchor, not promotion.

## Half-budget learning and efficiency

Each method used the selected discount-0.9/entropy-0.003 recipe and 12,492,800 transitions per seed (12.5M requested). Compare these variants with their own half-budget control; comparisons to full-budget results are budget-unmatched.

| Method | Mean success | Mean trainer minutes |
|---|---:|---:|
| short-entropy003-control | 12.2% | 11.5 |
| short-entropy003-lr2 | 4.7% | 10.7 |
| short-entropy003-epochs4 | 2.9% | 10.3 |

Doubling the learning rate did not improve success. Four update epochs reduced trainer time but degraded success beyond the allowed tolerance. Neither qualified for the conditional full-budget verification phase. The selected recipe did not retain performance at half the transition budget. This does not establish that all possible faster-learning methods fail.

## Completion and preservation

The six-run discount sweep and 21 additional overnight scientific runs completed without reported execution errors: 27 runs total, excluding smoke tests and the preexisting official baselines. The overnight controller stopped under its plateau rule after the half-budget phase.

Runpod read-back confirms Pod `2c1h36dlfn0t1f` is **EXITED**. The independent supervisor completed backup and stop. Local archive SHA256 matches the supervisor record: `66924758a2353527c2b9c7724a819b812cb096c2810d024f6df5e468f3eb1b99`.

Runpod billing read for this Pod on 2026-10-27 reports $5.999295849120244 GPU compute and $0.07060185237787664 Pod disk, $6.069897701498121 total. This is Pod-only billing; network volume storage is separate.

Backup: `.runtime/overnight-20261027-results.tar.gz`. Extracted safe JSON reports: `runs/stackcube-research-20261027`. Original PickCube and StackCube baseline artifacts remain separate.

Before further adaptive development tuning, the selected candidate merits an independent confirmation evaluation across its three training seeds. Remaining hypotheses include a gamma/entropy interaction, entropy annealing, and training-only placement/release shaping after inspecting updated rollout diagnostics. Vision should receive its own baseline and versioned observation interface.
