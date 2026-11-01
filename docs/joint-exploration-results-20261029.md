# Predicted joint-exploration benefit failed; its coordinate control learned more

The preregistered joint-exploration gate failed. Both fresh seed-1 learners
completed exactly102,400 transitions and received one frozen development score.
The unexpected early-stage gain came from coordinatewise exploration, not the
predicted joint-vector intervention. Neither policy stacked a cube.

| Behavior collector | Grasp /128 | Lift /128 | Transport /128 | Final stack /128 | Child process seconds |
|---|---:|---:|---:|---:|---:|
| Cached ordinary low-alpha SAC |64 |1 |0 |0 |81.12s checkpoint timestamp |
| Joint-vector10% uniform mixture |64 |7 |0 |0 |84.374 |
| Coordinatewise10% uniform mixture |83 |34 |10 |0 |89.241 |

The cached reference was not retrained or rescored. The new policy comparisons
share6,088 critic /2,029 actor and temperature updates, unchanged low-alpha SAC,
task/reward/horizon/action interfaces and safe deterministic evaluator. Placement,
release, contact-free placement and ten-step stability were zero for both new
policies. Times for new rows include imports and final exports; the cached time
is the saved checkpoint timestamp, not a complete process measurement.

The joint policy's7 lifts meet the <=8 weak-sufficiency threshold. It also fails
the required>=32 lifts and lead of>=20 above its control. Do not reverse the
primary hypothesis or call this a successful joint-exploration experiment.
Coordinate's34 lifts are an **unexpected, one-training-seed signal** relative to
the cached reference's1 lift. This is not confirmation, improved stacking,
10x sample efficiency, a tuned-SAC-versus-PPO verdict or a promotion.

## What was actually controlled

Both collectors have identical conditional coordinate marginals at any common
fixed actor/state. Their learned policies and visited states diverged. The initial
21 serialized actor/critic/log-alpha tensors matched each other and the cached
reference exactly. Saved CPU **and CUDA global RNG states** also matched at the
initial and102.4k checkpoints. This verifies the intended random-draw schedule;
it does not promise bit-identical physical trajectories.

Both used97,344 post-warmup policy proposals and778,752 independent coin plus
778,752 uniform scalar draws. Joint replaced all eight coordinates in9,716
actions. Coordinate replaced at least one in55,193 actions and all eight in0;
coordinate replacement counts were9,716/9,817/9,838/9,630/9,652/9,604/9,728/9,670.
The first-coordinate count was identical as expected from shared random draws.
Unmodified proposals/log densities, executed actions and masks were recorded
separately; executed mixtures were never passed to SAC as policy-density samples.

The preceding coverage gate predicted a median joint-command proxy advantage,
but that did not translate into better learning. It tested one-step target FK,
not physical contact, orientation retention, temporal behavior or critic learning.
The two mixtures also differ in whole-vector clean-step frequency; there is no
unique-mediator conclusion. The analytical rare/common negative control had
already shown that shared broad exploration can be worse. Further explanation
must be tested, not retrofitted as a successful original prediction.

The last horizon-aligned64 training episodes contained1 observed lift under
joint behavior and15 under coordinate behavior. Those are stochastic episodes
with changing weights and exploration, not frozen benchmark estimates.

## Preserved execution issue and full accounting

Both workers completed normally. The in-cloud bounded Volume API read returned
partially padded JSON bytes, producing Unicode decoding errors before either
scoring branch. The original failed coordinator summary remains unchanged.
External API reads hash-verified both full candidates; a reviewed score-only
handoff validated both initial/final policies and completed exactly the two
originally planned scores. No learner, endpoint or evaluation was repeated.
`reconciled-summary.json` links the original digest/errors and durable GPU IDs.

Additional training204,800 transitions,173.615 combined child seconds;12,176
critic and4,058 actor/temperature optimizer calls. Independent development
12,800 transitions/256 episodes,65.697 scorer-process seconds; zero confirmation.
All collector overhead is inside the measured trainer clocks. No diagnostic,
teacher or analytic-controller data entered training. Candidate workers could
not access the evaluator/private seeds/network or Modal credentials.

Archive:`runs/joint-exploration-20261029/`,44 verified candidate/score files,
28,611,299 bytes, plus inventories, exact sources and `production-parity.json`.
Both apps were independently read back STOPPED/zero tasks before closing their
shared$2 reservation. Reported campaign increment$.16830462; total session
$14.24728847, remaining$85.75271153, no active reservations. Metering may lag.
Scientific ledger70 completed training runs/38 recorded variant families;
historical PickCube/PPO and all failed studies remain unchanged.

## Reassessment

The complete-task question remains unanswered: PPO reliably solves StackCube,
whereas these SAC screens have not shown learned stacking. A useful next case
for review is one fresh-seed, fixed10-minute learning-curve comparison of ordinary
low-alpha SAC versus the coordinatewise collector. It would test whether the
unexpected early lift/transport gain persists and becomes stacking, with matched
step checkpoints and separately reported wall-time censoring. No such follow-up
is launched or implied by the failed joint gate itself.
