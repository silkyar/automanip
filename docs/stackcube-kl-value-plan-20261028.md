# Fixed from-scratch KL/value completion pair

The target is reducing new training experience toward the independently confirmed
reference near96% success. Two fresh seed1 runs compare ordinary strong PPO with
critic continuation after actor KL stopping. This is a known policy/value update
separation mechanism, not a novel algorithm or full PPG. No teacher, checkpoint,
demonstrations, assisted initialization, shaping or new model is permitted.

Source handoff: `research/discovery/from-scratch-kl-value-v2/`.

- `scratch-kl-stop-control`: `de067415f6895f7b4fcc534cff20fe9cd628fef6faae3846c3b49687ff526632`.
- `scratch-kl-stop-critic`: `8784f4d7aeb268a4794a59c509b761f740fffdd5932b0e6dd364ec4157c9228b`.
- Patch/private-RNG source: `b817a73a8c39d506ab8cf37970b64685488986e6e5a0ed62738f2b160568768a`.

Both retain the reviewed full-horizon public StackCube state48/action8/controller,
reward/physics,1,024 environments,50-step rollouts, gamma.9, entropy.003, LR.0003,
8epochs,32minibatches, original value MSE, gradient norm limit.5 and target KL.1.
Control keeps ordinary early break. When that same KL threshold is crossed, the
candidate stops actor updates and fits only the existing critic on remaining
minibatches of the **same eight configured epochs**, including the stop minibatch.
GAE targets stay detached/fixed; they are not recomputed. Critic-only clipping
uses critic parameters; existing joint clipping stays unchanged. Actor gradients
are None, preventing Adam momentum drift. Source restores the global NumPy state
at the stop point so suffix work cannot consume future policy minibatch RNG.

Root reviewed the source/schema and CPU tests: forced-stop actor mean/logstd,
Adam moments/step counters, action ratios/entropy and NumPy/Torch RNG match the
control; no-KL case matches all test updates/parameters bitwise. This is source/
CPU evidence, not a GPU success result. Biased value targets can make extra fitting
harmful; reduced value loss alone cannot pass the scientific gate.

Each run requests **10,240,000 new transitions** (200×51,200), or one original
**600-second dispatch lease**, whichever ends first. Reserve20s for saving;
wrapper660s plus30s startup. Queue and setup consume the lease; no refresh after
preemption/retry. At most two dedicated L40S trainers run concurrently. Record
seed1, actual transitions, runtime/setup/queue and source/native/safe-artifact
hashes. Initialization is random under the reproducible reviewed recipe.

Preselected frozen128-episode development scores at **5,120,000 (`ckpt_101.pt`)
and final actual steps**: max four unique queries across the pair. Merge coincident
steps and reuse identical actor hashes; reject future/missing expected checkpoints.
Missing later points are censored. No extra matching evaluations, confirmation,
private holdout, extra seeds or GPU smoke. Task/scorer/interfaces/seeds stay frozen;
only the existing restricted converter decodes native weights. Preserve all native
checkpoints/events/logs and complete safe artifacts/development reports.

Every rollout emits `AUTOMANIP_KL_VALUE_UPDATES` plus numeric `audit/kl_value/*`
traces. Validate exact source identities, role counters, cumulative sums,
first-stop epoch/minibatch, observed KL, configured8×32 limits, deadline skips,
critic minibatch rows/time and unchanged actor/target/data flags. Actor=joint;
critic=joint+critic-only; total unique optimizer calls=joint+critic-only. Legacy
`audit/optimizer_steps` stays **joint/actor only**. Record its value separately;
primary `optimizer_step_count` must be measured total calls. More updates are
compute, not extra simulator data. Built-in training evaluations and frozen scoring
experience/runtime remain separate from training transitions.

Outcome rule before results: equal final actual new steps are required for a final
paired comparison. A separately planned follow-up is warranted only if the
candidate gains **at least3 final paired successes** over control and has no
5.12M regression larger than3 successes. If an expected common point is absent,
this gate is not established. Report censoring and actual steps/time; do not
extrapolate a sample-efficiency win or cherry-pick another checkpoint. No automatic
retry, expansion or confirmation. Reliable step reduction requires later independent
training seeds and confirmation under a new admission.

Minimal adapter derives from reviewed precision/CAPG flow: separate trainer output
Volumes/once-keys, restricted networking/RPC, readonly PhysX cache, trusted CPU
coordinator and unchanged frozen converter/evaluator. No data/teacher mount or
candidate Python enters the evaluator. New contracts freeze exact source/CLI and
role accounting; no orchestration framework is introduced.

Proposed **$3** envelope within the one shared$100 cap, requiring one separately
recorded reservation before launch:1,980 GPU-container seconds (two660s wrappers
+startup, four120s evaluations+startup), four CPU conversions and3,600s coordinator,
**$1.2655518** resource subtotal /**$1.7344482** reserve. Report actual meter/lag and
all setup/failures. This plan is default denied until root's final adapter/source/
scientific review and paid admission; it changes no current BC/AMP run or allowance.
