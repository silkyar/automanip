# StackCube official PPO reference

[Completed results, diagnostic videos and proposed experiments](stackcube-results-20261026.md).

This is a separate benchmark from PickCube. PickCube's task fingerprint,
reference checkpoint files and results are preserved. No autonomous optimization
or training-reward changes are part of this batch.

## Reference definition

The first StackCube run uses the unchanged `ppo.py` and 25M-step state example
from ManiSkill v3.0.0b21, commit
`6226567ce326625fa6eacf9aeb3f4008e4339f7e`. Copies and attribution are under
`third_party/maniskill` and `THIRD_PARTY_NOTICES.md`.

The release also contains a distinct 50M-step `ppo_fast.py` benchmark in
`baselines.sh`; this batch follows the requested 25M-step classic PPO example
from `examples.sh`, not that published fast-PPO recipe.

Command settings: StackCube-v1, 1,024 environments, 50 steps per rollout, eight
update epochs, 32 minibatches, and training seeds 1, 2 and 3. Algorithm parameters
not specified by that command retain the official script defaults. Video capture
and external W&B tracking are disabled as operational flags; TensorBoard logging
and official training/evaluation logic remain unchanged.

The official script floors its iteration count. A request for 25,000,000 steps
therefore produces 488 × 1,024 × 50 = **24,985,600 transitions** per seed. It is
step-budgeted, unlike the previous ten-minute PickCube batch.

The task's default `panda_wristcam` robot, `pd_joint_delta_pos` controller,
48 state observations, eight actions and 50-step horizon are frozen separately.
The attached wrist camera does not make this a visual policy: observations are
state vectors. Official StackCube rewards and task source are unmodified; the
adapter verifies the task source hash before evaluation.

## Independent scoring and diagnostics

The evaluator uses our existing rollout/result contracts, safe policy loader,
and a separately frozen StackCube adapter. PickCube retains its seven-action
default; StackCube explicitly requests eight-action validation.

Evaluation independently reads cube poses, velocities and gripper forces.
Stock success requires the upper cube to satisfy the official XY/height
alignment thresholds, linear speed <= 0.01 m/s, angular speed <= 0.5 rad/s,
and the official non-grasping condition. Both final and ever-success are reported.

Frozen protocol: `benchmarks/stackcube-v1/protocol.json`.
Task fingerprint:
`b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f`.
There are 128 fixed development episodes and 512 separate confirmation episodes.
Confirmation episodes are independent from development; they are not a private
holdout service. No private-final-holdout result is claimed.

Diagnostic definitions are fixed before the batch:

| Diagnostic | Definition |
| --- | --- |
| Grasp | Official bilateral finger-force/angle grasp condition ever observed |
| Lift | Upper-cube center at least 3 cm above its reset height |
| Transport | Lifted and horizontal distance to the lower cube reduced by at least 3 cm |
| Placement | Official on-top geometry ever observed |
| Release | On top and no longer officially grasped |
| Contact-free placement | On top and both finger-force norms <= 1e-6 N |
| Stability | Ten consecutive control steps satisfying stock success |

Non-grasping is not the same as zero finger contact. Stability and contact-free
placement are extra diagnostics, not changes to the stock success criterion.
Failure categories describe the first unobserved stage; they are observational
summaries and do not establish a physical cause by themselves.

The unchanged trainer writes native tensor checkpoints. A separate training-side
converter uses restricted `weights_only=True` loading, checks the exact official
actor/critic tensor schema, and exports the actor to our validated JSON format.
The trusted evaluator never imports upstream training code or loads those native
checkpoint files. Vectorized NumPy policy inference is checked against the
existing scalar reference implementation.

Independent development curves are measured at 0, 5.12M, 10.24M, 15.36M, 20.48M,
and final steps, using preselected saved checkpoints. Both development and
confirmation evaluation use the final checkpoint for the final score; no
best-checkpoint selection is performed. Official TensorBoard curves are also
exported as JSON and CSV. Process wall time, simulator rollout time and optimizer
time are recorded separately; process time includes official in-loop evaluation.

The offline report also plots independent development success against elapsed
training-process time. Intermediate checkpoint times come from the official
TensorBoard step timestamps; final time comes from the supervised process timer,
which has up to five seconds of polling granularity. A threshold crossing is only
the first observed crossing at these checkpoints, not the exact convergence time.

## Interpreting the earlier PickCube timing

The PickCube reference trained for a fixed 600 seconds per seed and evaluated the
final policies. It established 97.7–99.2% development success after ten minutes;
it did not measure the earliest time to that success rate. Do not describe this
as evidence that ten minutes was necessary.

The [ManiSkill3 paper, section IV-C](https://arxiv.org/html/2410.00425v2#S4.SS3)
reports near-100% state-based PickCube success after about one minute on an RTX
4090 with delta joint-position control and hyperparameters tuned for wall time.
That means success over evaluation episodes, not merely one successful pick.
Our earlier run used an L4, 512 environments and an adapted PPO trainer with
`pd_ee_delta_pose`, which includes inverse kinematics. Hardware and recipe differ,
and we have not reproduced or profiled their one-minute result. The quoted time
also belongs to PickCube, not StackCube.

## Execution and shutdown

The original PickCube Pod could not resume because its host GPU was occupied.
A new Secure Cloud L4 Pod `sptbqb2mwe2ed7` was created in EU-RO-1 at the live catalog
rate of $0.59/hour, attaching existing standard volume `zvx3i1sjnx`.
The official Runpod Python 3.11 image is reused, with a new pinned virtual
environment and the additional pinned TensorBoard dependency.

Remote workspace: `/workspace/automanip-stackcube-20261026`.
Local watchdog state: `.runtime/stackcube-diagnostics-supervisor.json`.
The standalone watchdog holds the local API key outside the GPU Pod, monitors
the exact Pod ID, attempts a bounded archive download on batch completion,
checks its SHA-256, stops compute and confirms `EXITED`. After the requested
post-baseline diagnostics were added, its completion trigger became
`diagnostics.exit`; a failed setup/training batch also triggers shutdown.
It still enforces the original
eight-hour runtime ceiling and a $5 rate-based envelope, using $0.60/hour as a
conservative GPU-plus-storage rate. Actual final cost comes from billing reads.

The watchdog is detached from the chat and runs under `caffeinate` to prevent
idle sleep. It still depends on this local machine remaining powered, connected
and able to reach Runpod; it is not an always-on managed cloud service. It does
not grant the trainer cloud-control credentials. Filesystem/identity isolation
against a malicious research agent remains a separate milestone.

Local validation: 44 CPU tests, including the later diagnostic-selection and
completion-trigger checks; real CPU seeded reset and
independent/upstream stock-rule agreement; exact upstream source hashes; safe
checkpoint conversion; scalar/vectorized inference parity; supervisor deadline,
cost, credential parsing and exact-Pod stop-request tests. GPU setup, smoke,
training and final scores are recorded after actual execution.

## Files to inspect

| File | Responsibility |
| --- | --- |
| `third_party/maniskill/ppo.py` | Unchanged official PPO trainer actually used for StackCube |
| `scripts/run_stackcube_baseline.py` | Three-seed launcher, provenance, curves, export and independent evaluation |
| `src/automanip/stackcube.py` | Frozen StackCube interface, seed sets, diagnostic definitions and integrity checks |
| `src/automanip/stackcube_adapter.py` | Evaluator-owned simulator state reads and stock success calculation |
| `scripts/evaluate_stackcube.py` | Safe actor loading and development/confirmation evaluation |
| `scripts/export_official_policy.py` | Training-side restricted checkpoint conversion to actor-only JSON |
| `scripts/trace_stackcube.py` | Development-only chronological reward/action traces and videos |
| `scripts/report_stackcube.py` | Offline plots and descriptive replay analysis |
| `scripts/supervise_pod.py` | Detached local backup and exact-Pod shutdown monitor |

`train.py` remains the adapted PickCube trainer; StackCube's first reference uses
the unchanged upstream trainer to satisfy the official-baseline requirement.

## Stock reward and post-baseline diagnostics

The stock dense reward uses ordered overrides, then divides the selected value
by eight. The branch values are not all added together:

| Active branch | Dense reward before normalization |
| --- | --- |
| Reaching | `2 * (1 - tanh(5 * distance(tcp, cubeA)))` |
| Cube A grasped | `4 + (1 - tanh(5 * distance(cubeA, stacking_pose)))` |
| Cube A on top of B | `6 + (ungrasp_reward + static_reward) / 2` |
| Stock success | `8` |

The stacking pose is above B by 4 cm. Lifting is encouraged indirectly by this
3D target; there is no separate lift-height term. The ungrasp reward uses
normalized gripper width while grasped and becomes one once no longer grasped.
The static reward is `1 - tanh(10 * linear_speed + angular_speed)`. The on-top
branch already rewards opening and stillness, so an extra release/stability
reward needs a specific hypothesis rather than assuming those incentives are
absent. A nearly aligned, held cube can obtain substantial reward without stock
success; that observation alone does not prove a local optimum.

`scripts/trace_stackcube.py` runs only after all three baseline seeds finish.
It replays development episodes, never confirmation episodes. Every development
episode gets step-by-step normalized actions, stage flags, cube positions and
stock reward-component traces, plus a chronological stage-transition summary.
These are additional descriptive replays; the original frozen reports remain
the authoritative benchmark scores.

Video selection takes the first successful development episode per training
seed, when available, and the first example of four different failure stages.
This produces up to seven illustrative videos, with detailed state/force/action
traces and plots. Selection is descriptive, not a representative sample or a
new performance estimate. Each replay retains the original eight-environment
seed batch. Replay/benchmark diagnostic agreement is logged rather than assumed.
The reward reconstruction is checked against the untouched official task method
at every recorded step, and generated videos must decode with 51 frames.
