# Does search improvement survive policy compression?

The relative-gripper search passed its primary coverage/fidelity screens: 89/128
source states have a successful current-relative continuation versus 1 absolute
and 19 anchor-relative. This is an assisted training oracle, not a policy score.

Register a minimal supervised student and an unbranched control before fitting or
fresh assessment. Keep the inherited actor trunk and fit only its last affine
layer, exporting the unchanged safe MLP artifact schema. No clock, success flag,
geometry rule, controller wrapper or planner enters the student. Input is still
48 observations; output still eight original normalized commands.

Control collection: 18 independent batches x 64 episodes x 50 steps per parent,
115,200 transitions across both parents. Use the same current-relative proposal
distribution and step 30 transition, but one sampled continuation per new episode,
with no comparison between futures. This equals the entire coordinate study's
new physics, including its diagnostic arm/repeats, and gives the control more
distinct source episodes. Report the search's additional preceding 229,120
diagnostic/search interactions separately; no from-scratch or 10 x claim. Parent
8.192 M prefixes and full historical acquisition remain explicit assistance.

Both methods imitate successful actual continuations and rehearse the parent's
raw action means on early observations. Split source episodes deterministically
75/25 using fixed seeds. Fit each student on 384 teacher rows and 1,024 rehearsal
rows sampled from fitting episodes; report actual shortages rather than invent
targets. Validate on up to 64 teacher and 256 rehearsal rows from other episodes.
No hyperparameter search or refit after observing policy assessment.

The single closed-form readout update minimizes mean teacher raw-action squared
error +4 x mean rehearsal raw-output change +0.001 x squared weight change.
This is one 257 dimensional linear solve per parent/method. Count every feature
forward and CPU second. A separable-action toy verifies fitting and retention;
an aliased-state negative control exposes their unavoidable conflict. Safe JSON
export/reload is tested. Offline losses are diagnostics, not a veto or success
claim; actual closed-loop behavior is required.

Assess original deterministic parent, search-distilled MLP and unbranched-distilled
MLP on 512 fresh public episodes per parent/variant (153,600 physics transitions
total). Use the frozen evaluator rollout/success kernel in a separate scoring
worker accepting only safe MLP artifacts, with no learner data or proposal code.
Every policy acts from reset for all 50 steps, no planner or scheduled correction.
Count all assigned failures. Fresh 962 series seeds are not reused for fitting.

Primary learning screen: search student gains>=10 percentage points over original
parent, positive 97.5%paired block-bootstrap lower bound, with >=95%ever-grasp in
each fixed parent. For a search-specific benefit it must additionally gain>=5
points over the equal-physics unbranched student with a positive 97.5%lower bound.
These are separate criteria: if both students improve but search does not beat
control, preserve the learning result without crediting branching. Two acquired
parents are not a population replication or held-out-task validation. No private
confirmation/final query and no baseline replacement without later replication.

Projected cumulative branching investigation:344,320 prior +115,200 control
collection +153,600 assessment =613,120 of 1 M. Zero PPO/GPU training updates; four
CPU readout fits. If this fixed learner fails, keep the failure and distinguish
representation/retention limitations from insufficient planner coverage before
another method. Historical benchmark/training rows remain untouched.
