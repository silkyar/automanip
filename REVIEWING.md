# AutoManip research review snapshot — 2026-10-29

Start with these files, in order:

1. `PROJECT_SPEC.md`, `AGENTS.md`, and `research/AUTOMANIP_RESEARCH_MANDATE.md`: intent, frozen evaluation rules, and research mandate. The initial MVP phase text predates the historical GPU experiments; use dated results for current status. Review only; do not launch cloud jobs or change evaluation rules.
2. `docs/experiment-ledger.md` and `docs/experiment-runs.json`: all72 completed scientific training runs,40 recorded variant families, task, seed, training time, transitions, change and measured success. Setup, CPU probes and evaluation are separate.
3. `docs/research-strategy-diagnosis-20261028.md`: why endpoint tuning did not achieve the sample-efficiency objective, confounded interventions, and unresolved mechanisms. `docs/literature/survey-20261028.md` links the primary-literature reading.
4. `docs/first-update-probe-results-20261028.md`: assisted BC→PPO collapse and the exact first-update mechanism probe, including its inconclusive small-step result. Assistance is fully charged and is not from-scratch progress.
5. `docs/joint-exploration-results-20261029.md` and `docs/full-stacking-curve-results-20261029.md`: most recent exploration prediction failed; its control improved intermediate skills, but the fresh-seed ten-minute comparison produced zero completed stacks at every saved checkpoint. Acquisition and regression are both retained.
6. `publication/stackcube-ppo/README.md`, `ppo.py`, `official.json`, `recipe.json`, and `recipe.diff`: minimal PPO extraction and the concrete discount/entropy/full-horizon changes behind the reliable stacking recipe.
7. `research/discovery/sac_baseline.py`, `replay_learner_lowalpha.py`, `joint_exploration.py`, and `replay_robot_full_curve.py`: latest SAC actor/critic, collector and training code. Exact per-run copies and hashes are under the relevant `runs/*/source/` directories.
8. `src/automanip/policy_io.py`, `evaluator.py`, `stackcube.py`, `stackcube_adapter.py`, and `deploy/modal/evaluate.py`: safe actor validation, trusted metrics/task, independent scorer and once-only scoring. `tests/` contains contract/isolation checks; `deploy/modal/` and `deploy/runpod/` contain execution infrastructure.

## What the evidence currently establishes

- PickCube:97.7–99.2% development after600 seconds across three seeds; the stock pick metric does not require release.
- StackCube:25M recipe94.08% independent confirmation;50M recipe95.90% across three training seeds. One historical strong PPO control scored98/128=76.56% at10.24M transitions in572.61 child-process seconds on L40S. These are different budgets, not interchangeable headlines.
- Latest full-task SAC comparison:two fresh seed2 runs,586.81/587.05 seconds,783,936/801,472 transitions, zero final stacks at every declared checkpoint. Coordinate exploration lifted58/128 at512k and24/128 at the terminal checkpoint. No successful sample-efficiency claim.
- The objective of at least10× fewer interactions to80–90% against strong matched baselines remains unmet. One development seed or a diagnostic subskill gain cannot establish it.
- `research/discovery/history-motion-plan-20261029.json` and `history_motion_prediction.py` are a reviewed but **unrun, default-denied CPU diagnostic** in this snapshot. They are not a trained method or admission for another GPU experiment.

## Snapshot contents and exclusions

This branch includes project source, tests, dependency pins, scientific reports, small experiment records/source copies, plots/videos and the standalone PPO extraction. `UPLOAD_MANIFEST.json` records the copied files and hashes. Main was not changed.

Credentials, local virtual environments/interpreters/caches, private account configuration, native training checkpoints, bulk `runs/**/weights.json`, bulk trajectory NPZ files and TensorBoard event files remain local. Small existing research fixtures already in the source tree are retained. Inventory manifests can therefore refer to bulk files intentionally absent from Git. Request those separately if needed; do not infer a failed hash check from an omitted file.

Only specifically selected nonsecret scientific ledger receipts from `.runtime/` are included. The legacy settling tarball is a small derived container with its original scientific summary member only; raw summary bytes and the generator's member digest are unchanged. The original full archive remains local.

The generated ledger is authoritative for the recorded history. Rebuild with `python scripts/build_experiment_ledger.py`; this does not launch training or scoring. Runtime plans may retain historical admitted flags and resource IDs; they are records, not authorization to execute them.

A useful review question: Which causal hypothesis best explains acquisition followed by regression and weak composition, what evidence contradicts it, and what is the cheapest experiment that could falsify it? Keep algorithm conclusions narrow and charge all experience, assistance and compute.
