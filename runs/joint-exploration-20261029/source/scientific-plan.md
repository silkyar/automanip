# Does joint broad exploration help learn the arm–gripper combination?

Registered before training. The physical intervention rescued lifting with
closure+upward motion, not either alone; the fixed-state matched-marginal coverage
gate passed (median proxy9.62x; working-MC-bound check6.01x). These are assisted
mechanics/command probabilities, **not learning gains**. All discovery data,
its own-policy prefix, structural kinematics and compute remain in the ledger.
No diagnostic state/action/controller/teacher enters this fresh learner.

## Competing explanations and intervention

H_joint: simultaneous broad exploration of action coordinates more often
discovers useful arm–gripper combinations than distributing the same marginal
exploration independently over coordinates. H_null: this proxy does not improve
learning, and disruption, value fitting, temporal persistence or other issues
dominate. The mechanism can help a rare/rare action conjunction but hurt a
rare/common conjunction; its analytical negative control is already recorded.

After the identical5,056-transition uniform warmup, change **only the behavior
collector**. Joint: with epsilon.1 sample the entire8-vector uniformly, otherwise
use the ordinary SAC proposal. Coordinate: independently replace each coordinate
with uniform probability.1. Their conditional coordinate distributions match
at any common fixed actor/state; individual learned actors/state distributions
will diverge. Both retain the original low-alpha gamma.99 SAC learner, target
policy samples/log probabilities, temperature update and safe deterministic
export. No new objective, actor architecture, data source or task interface.

Joint exploration also leaves90% of steps wholly unperturbed; coordinatewise
exploration leaves`.9^8≈43%`. These are inseparable consequences of the dependence
change. A learning difference would not uniquely identify rare-event discovery
versus uninterrupted exploitation; record masks and clean/broad run lengths.
Run lengths are diagnostic summaries of the retained last50step windows at51.2k
and102.4k, not the full history; cumulative collector counts are logged every5,120
transitions and at finalization. These checkpoints contain64 horizon-aligned
episodes each, with behavior weights changing during collection.
This is not temporal-memory exploration and contains no task-stage oracle.

Always call the original actor sampler once per learned-collection step in both
arms. Draw equally many coins/uniform values using a separate seeded generator;
joint selection broadcasts the first coin, coordinate selection uses all.
Original warmup/global actor, replay and learner RNG consumption remain unchanged.
Record proposal actions/log densities separately from the **executed** mixture
actions and masks. Never call the proposal density a behavior density or use it
in learning. SAC updates use their own unchanged target-policy probabilities.

## Cheap gates and primary comparison

Before admission: exact epsilon0 original-sampler/global-RNG parity on synthetic
inputs; dedicated-generator reproducibility, bounds/mask semantics, unchanged
initial actor/critic/target tensors and exact core-source diff; fixed learner
update/terminal tests; source/dependency/budget review by an independent reviewer.
No GPU smoke is needed simply to count as another experiment.

Two fresh seed-1 runs, exactly102,400 transitions/150 child seconds maximum each,
same64envs, replay/optimizer/minibatch/update budgets. One frozen development
score at exactly102.4k per unique policy; no later endpoint, extension or
confirmation. Reuse the untouched low-alpha reference (grasp64/lift1/stack0 at
102.4k) without training/rescoring. Beating a damaged coordinate control alone
does not establish an improvement over that baseline.

Promising gate: joint grasp>=64/128 AND lift>=32/128 AND joint lift at least20
episodes above coordinate lift. This gates another reviewed investigation only;
no automatic promotion/replication. Joint lift<=8 weakens its sufficiency at
this budget. Other counts are partial/inconclusive. If either primary is missing,
retain censoring/failure, never substitute a later or repeated run. Always report
all stages, final stacks, wall time, optimizer work, collector fractions and
training curves. This is a one-seed mechanism test, not10x stacking efficiency.

## Execution and isolation

One shared$2 reservation within the existing total$100Modal cap. Up to twoL40S
workers,150s hard children/210s functions+30startup; one900s CPU coordinator;
at most two120s frozen development functions+30startup. Original no-new-Runpod
constraint remains. Claims/leases prevent repeat scientific jobs and GPU scores.
Use bounded Volume API reads for completed receipts/artifacts, avoiding the
previous mounted-directory visibility race. Retain read errors; a repeated
file read is not a repeated training/scoring job. Candidate files remain data,
accepted only through the frozen safe loader in the trusted coordinator.

Workers receive only their source/plan, official task runtime and read-only
physics cache, with no pretrained policy, evaluator/private seeds, network or
Modal access. Archive every attempt and score, preserve errors and source hashes,
verify stopped/zero tasks and then close the reservation against actual billing.
Independent scientific/code review precedes root admission.
