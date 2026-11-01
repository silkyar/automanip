# Feature-value learning: negative result

Both the state-only and clock-aware value baselines scored **0/128 at every observed checkpoint in all three training seeds**. They failed the registered sample-efficiency screen. The frozen StackCube task, actor interface and historical benchmarks remain unchanged.

This followed a controlled CPU finding: scaling current actor features by remaining discounted horizon reduced whole-trajectory policy-gradient covariance, including all 146,192 mean and log-standard-deviation parameters, in 12 retained training-data cells. Fixed shuffled-clock and state-only controls were included. This established an estimator property on those batches, not faster policy learning.

The online experiment compared two otherwise identical from-scratch procedures: finite 50-step Monte Carlo returns, actor-only PPO, and episode-cross-fitted linear values on 257 frozen current actor features. Only one procedure scaled features by `1 - gamma ** remaining_steps`. The actor received the same 48 observations and produced the same eight actions. Both removed neural-critic training; comparisons with historical GAE therefore bundle several changes. The state-versus-clock comparison isolates the feature scale.

| Method | Seed | Actual transitions | Trainer seconds | PPO + value-fit seconds | Final success |
|---|---:|---:|---:|---:|---:|
| State features | 1 | 9,676,800 | 595.59 | 177.81 | 0/128 |
| Clock features | 1 | 9,932,800 | 596.48 | 180.28 | 0/128 |
| State features | 2 | 10,035,200 | 595.41 | 178.09 | 0/128 |
| Clock features | 2 | 10,188,800 | 594.52 | 181.80 | 0/128 |
| State features | 3 | 10,240,000 | 536.00 | 141.69 | 0/128 |
| Clock features | 3 | 10,240,000 | 596.72 | 179.46 | 0/128 |

Each run used L40S, four CPUs, 16 GB RAM and dual ceilings of 600 trainer seconds and 10.24M transitions. Linear fitting consumed 21.02–29.21 seconds per run, included above. There were 60,313,600 actual training transitions and 307,200 independent scoring transitions. Forty-eight checkpoint assessments completed without execution errors, using paired public development episodes and the frozen evaluator. No confirmation or final holdout was queried. T80 and T90 remain censored; there is no measured speedup to report.

The result rules out this particular procedure within its budget. It does not rule out variance reduction generally, and does not establish why finite-MC training failed. Further parameter sweeps are not justified by this finding. A separate diagnostic will test whether the failed GAE policies have acquired placement but lack a reliable corrective release sequence; manual intervention cannot count as a trained-policy result.

The [registered protocol](../cumulative/experiments/feature-value-v1.json) and [complete summary and learning curves](../cumulative/results/feature-value-20261030.json) contain exact counts. Local immutable archives retain 2,488 training files and 18 scoring files, with SHA manifests, under `artifacts/cumulative/feature-value-20261030/`. Blind traces cover only the first 64 training lanes; full critic states, all per-rollout actors and complete 1,024-lane trajectories were not retained. These assets cannot support an exact alternative-training replay.

CPU checks actually run: eight feature-value fitting, source-preparation and literal generated-update tests passed. A prior combined 32-test check covered related diagnostics. These checks are separate from the six actual GPU training runs and six frozen scoring jobs above.
