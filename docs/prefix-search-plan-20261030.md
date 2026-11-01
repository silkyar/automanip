# Prefix-replay search before distillation

Direct state restoration failed. A no-op write changed subsequent motion, while
fresh-process capture-only trajectories matched exactly. Test an alternative:
reconstruct the same forward history in a fresh process for every candidate.
No state teleport, setter, rewind, shaped reward, learned value or supplied
release/hold option is used. All repeated prefixes count as new transitions.

Use two existing weak 8.192M parent checkpoints. They are explicitly acquired
assistance; both full 10.24M historical acquisition runs remain accounted for.
Generate 64 new training-only episodes per parent using registered 960-series
reset/noise seeds. All candidates share the same first 30 actor/Gaussian actions
and common suffix innovation tape. Each candidate continues to step 50.

The sixteen proposals are: candidate 0 continues the original stochastic policy;
candidates 1–8 apply an independently sampled, constant normalized action from
N(0,0.05²I) in all eight coordinates; candidates 9–15 add a constant N(0,0.25²I)
offset to the original Gaussian policy. All actions are clipped to [-1,1]. These
are fixed proposal settings, not a parameter sweep. The quiet-action distribution
is an **explicit action-space prior**. Joint deltas and the gripper's absolute
target have different semantics; no symmetry/general robotics principle is
claimed for this distribution. A single quiet candidate and an independently
assigned random candidate are controls for benefits supplied by the prior alone.

Every child is a fresh process, on the same pinned L40S stack and frozen task.
Repeat candidate 0 and quiet candidate 1 once per parent to measure fidelity for
both continuation types. Exact prefix hashes are descriptive; admission requires
all prefix states within unchanged registered physical tolerances, with identical
executed prefix commands and grasp/success flags. Repeated complete trajectories
must meet the same tolerances and have identical action tapes. Any fidelity
failure blocks training and is preserved, without tolerance relaxation.

Score candidates by actual **final stock success**, not ever success or shaped
reward. At each training state, preserve candidate 0 if it succeeds; otherwise
choose uniformly among successful candidates using a pre-fixed tie seed. No
successful candidate means no positive teacher target. This is an oracle over
training simulations, not a deployable policy or benchmark score.

The local label-coverage screen requires at least 64/128 source states to have
a successful continuation, pooled oracle coverage at least 15 percentage points
above the pre-fixed single quiet candidate, and no parent with negative oracle
gain. Report all candidate outcomes, equal-cost random proposal controls, stages,
setup times and physics. These descriptive training screens establish feasibility
only; selection among futures is not evidence of learned-policy improvement.

Costs: 18 fresh children ×64 lanes ×50 steps ×2 parents = **115,200 transitions**,
including the prefix in each branch and four full repeat checks. Zero optimizer
calls. Two bounded GPU functions, each <=300 seconds, no retries. Cumulative
branching investigations after completion: 229,120 transitions, leaving 770,880
under the user-approved one-million ceiling for subsequent fitting/assessment.

If fidelity and coverage pass, register a small student fit and fresh policy-only
assessment. Use the existing safe MLP architecture, original 48 observations and
8 actions. Preserve early behavior through rehearsal, test retention explicitly,
and compare against original policy plus an ordinary-rollout learning control.
No candidate success flag, geometry threshold, planner or time switch may be
inside the deployed MLP. No private confirmation/final set is used for this probe.
If coverage fails, investigate sparse terminal labels rather than tuning the
proposal scales repeatedly. Historical scores and failed experiments stay intact.
