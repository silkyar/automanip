# Research journal, October 31

The Grow–Plan–Distill pilot is closed negative. See the [scientific closeout](gpd-pilot-closeout-20261031.md) for comparisons and scope, and the [machine accounting](../cumulative/results/gpd-pilot-closeout-20261031.json) for all 984,320 transitions. No performance benchmark was promoted.

The causal chain was: relative proposal coordinates recovered successful training continuations; readout distillation failed to reproduce them reliably; removing rehearsal destroyed grasping; protecting executed rather than raw actions produced a small insufficient gain; extra student-state labels did not improve the fixed learner. CPU representation and normalization probes also failed their precision gate before robot scoring. Negative evidence was retained rather than selecting another learner against the same public scores.

Source-state replay checks passed only without simulator state writing. This implementation-specific limitation must be carried forward into any planning or certification claims. Oracle best-of-bank coverage must remain separate from ordinary policy-only performance and must count every attempted branch.

Current next step: a CPU-only test of a known hold-position mapping embedded in the existing actor architecture, followed by a decision about a narrowly controlled from-scratch experiment. The completed pilot will not be extended. The long-term 10× sample-efficiency objective remains unmet.

## Subsequent from-scratch test

The hold-coordinate CPU gate passed; a separately registered three-control, one-seed experiment completed. All conditions reached 4.096M transitions in 213–250 seconds. Every independent checkpoint scored 0/128. The screen failed and was not extended. See [complete results](gripper-hold-results-20261031.md), including a reporting omission: scored stage diagnostics were not persisted. Three real GPU training runs bring the ledger to 87 runs / 47 families; the original 84 rows and all 1,684 protected files remain unchanged. Both new apps are archived and stopped with zero tasks.

## Further local diagnosis

An exact-coordinate readout probe failed: target errors remained 11.98/9.86 mm. An exploratory native-stage audit found that 44.8%/50.7% of measured rehearsal fitting rows preserved closing commands in already on-top/grasped states. Removing those gripper labels improved corrective-action fit, but failed its joint comparison against time-matched random removal and retention requirements. No robot assessment followed. Six additional linear solves took about .31 CPU seconds total, with no new simulator interactions or policy exports. See [diagnosis and limitations](rehearsal-diagnosis-20261031.md).

The next methodological requirement is to distinguish preserving task competence from preserving every old action, while avoiding attribution to a stage rule when equivalent weakening of rehearsal explains the result. No successful cumulative mechanism or from-scratch efficiency gain has yet been established. Existing temporal-noise/history results and gradient logs were checked before entertaining those explanations again.
