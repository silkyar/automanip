# Exposed state restoration did not reproduce uninterrupted futures

The registered branching gate failed. **Zero of 24 replay batches passed for
either physical-only or augmented restoration.** Do not proceed to distillation
on the assumption of identical simulator futures. This is a simulator/interface
diagnosis, not evidence against planning as a general learning method.

Both existing weak parents generated 64 fresh training episodes. Snapshots at
steps 0, 15, 30 and 40 were replayed three times to step 50 using the exact original
float32 action tape. The augmented variant also restored exposed controller
buffers, joint drive targets/velocities, generalized forces, clocks and individual
random generators. It could not restore hidden solver/contact state.

Immediate restoration was accurate: maximum cube/TCP position component error
1.907 micrometres; quaternion rotation error 5.433e-7 rad; robot qpos/qvel and
cube linear/angular velocities matched exactly. No lane failed the registered
physical tolerances immediately after restore. Subsequent dynamics nevertheless
exceeded those tolerances in 1,533/1,536 replay-lane instances under each variant;
median first failure was one control step after the anchor. These instances
share original states and are not 1,536 independent training episodes.

Grasp or success flags differed during 57/1,536 physical-only and 52/1,536
augmented replay futures. Final success differed in one physical-only instance
and zero augmented instances. Almost all original episodes ended unsuccessfully,
so unchanged final binary outcomes do not establish useful planning fidelity.
The physical gate remains failed; no tolerance was relaxed.

Exploratory comparisons of the three restored repetitions with each other also
passed zero of 24 batches under each variant. At reset-state anchors, median
one-step qvel error was about 3e-7, but later anchors often showed millimetre
position changes and much larger velocity changes after one control interval.
That pattern motivates investigating state-write/contact-history effects; it
does not establish a hidden-cache, RNG or controller bug. A new narrow test
compares writing the **current** state back with capture-only controls in fresh
processes, without any rewind. [Followup protocol](../cumulative/experiments/branching-noop-v1.json).

Compute: **94,720 new environment transitions**, zero optimizer calls, worker
50.72 seconds, wrapper 53.57 seconds on one L40S. Two preceding implementation
attempts stopped before any environment step: index dtype mismatch, then the
upstream BatchedRNG wrapper's incompatibility with generic deepcopy. Both failed
attempts/logs remain archived, and both had zero transitions. Explicit generator
get_state/set_state and dtype-aware indexing passed CPU regression tests before
the completed attempt. Setup time remains separate from measured transitions.

Twenty-four selected CPU tests now pass (0.247 seconds), including seven branching
tests. The final trace analyzer reproduced worker decisions; Linux/macOS rotation
reductions differed by at most floating-point roundoff, checked at 1e-12 absolute
agreement without changing any scientific tolerance or Boolean decision.

All three apps were read back stopped with zero tasks. The successful execution
archive contains 55 remote files /19,710,431 bytes plus source/input provenance.
No confirmation, final holdout or optimizer was used; historical benchmarks and
the 84 scientific training records remain unchanged.

[Protocol](../cumulative/experiments/branching-validity-v1.json);
[analysis](../cumulative/results/branching-validity-20261030.json);
traces and exploratory analyses: `artifacts/cumulative/branching-validity-r3-20261030/`.

Upstream context: ManiSkill documents the exposed
[actor/articulation state schema](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/tutorials/custom_tasks/advanced.md)
and warns that [action replay can fail under simulation nondeterminism](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/datasets/replay.md).
Those current documents are context; this experiment used pinned 3.0.0b21 sources
and the existing PhysX binary, not an upgraded simulator.
