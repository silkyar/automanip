# Randomized experience predicts when a correction works

Two small outcome models predicted corrective success better than constant or time-only baselines on held-out training batches. This passes the registered predictive screen, **not a policy-performance or sample-efficiency test**. No robot policy weights were optimized.

The corpus contains4,096complete50steptraining episodes,204,800new transitions. Each of two existing8.192Msource actors supplied2,048episodes. Treatment was randomized independently of geometry and outcomes: continue the base actor, or switch at a random time10..44to zero arm delta and open gripper for the rest of the episode. This fixed corrective action is an explicit supplied prior. No geometric trigger or earlier public diagnostic trajectories entered fitting.

Among1,024correction episodes per source,529and421finished successful. The corresponding base groups scored1and0. These are training-corpus outcomes under randomized timing, not frozen benchmark scores or learned-selector results. The parent checkpoints came from completed10.24Mtraining runs. Minimal parent prefixes total16.384Mtransitions; actual completed parent training totals20.48M, already recorded historically.

| Source | Actor-feature Brier error | Constant | Decision-time bins | Shuffled-label head |
|---|---:|---:|---:|---:|
| Seed2 | .08948 | .12558 | .11784 | .15159 |
| Seed3 | .09008 | .12025 | .11293 | .14952 |

Lower Brier error is better. Each source used1,024fitting,512internal-calibration and512testepisodes, separated by whole vector batches. Two linear heads used existing257actor features; privileged cube velocity, rewards and future states were excluded. Four registered98.75%block-bootstrap intervals were positive, and both comparisons in each source met the absolute.02material improvement criterion. This does not establish calibrated success probabilities or a useful closed-loop trigger.

The raw head illustrates that limitation: it would trigger immediately in835/2,048and587/2,048recorded reset states, outside the time/state support of its corrective training data. An action-conditional leverage heuristic, using a fixed95thpercentile cutoff from unlabeled internal-calibration states, rejected all these reset states. This is a coverage heuristic rather than a confidence guarantee. It motivates one fresh comparison of base, random-time correction, raw learned selection and supported selection, with all weights and thresholds frozen before execution.

The pinned task already rewards ungrasping and staticity. Additional velocity diagnostics showed that most on-top static failures involved **both** excessive linear and angular velocity. The stored velocity vectors exactly reproduced every task static flag. Instantaneous angular velocity sometimes exceeded its threshold while endpoint rotation averaged over a control interval did not; those are different measurements and do not prove a simulator bug. All conclusions remain specific to the pinned task and success rule.

The collector ran in isolated trainer images containing safe parent actors, pinned dependencies and four collector/helper files, with no evaluator or benchmark seeds. It used zero learner updates and zero independent scoring queries. Native object-velocity reads were separately counted:409,600vector rows,6,400vectorized getter calls, excluded from learner features. Workers took68.94and64.73seconds; wrappers78.91and70.69seconds.

The archive contains74remote files /50,608,061bytes with hashes and source provenance. CPU model fitting used eight linear solves,4,096source-feature rows and26.88seconds while the behavior audit ran concurrently. The audit reconstructed all204,800executed action rows exactly in39.56seconds. Four collector/analysis tests and actual-parent sampling parity checks passed before launch.

Raw Gaussian proposals after the switch are **unused proposals**, not samples of executed corrective behavior. Executed corrections are deterministic point masses. The [data contract](../artifacts/cumulative/random-switch-training-20261030/data-contract-v1.json) records this distinction; these data cannot be fed into ordinary PPO using parent Gaussian likelihoods. The native corpus has not been imported into the generic cumulative episode store.

[Registered collection/analysis plan](../cumulative/experiments/random-switch-training-v1.json); [complete summary](../cumulative/results/random-switch-training-20261030.json). No main benchmark, historical run or private confirmation/final set was changed.
