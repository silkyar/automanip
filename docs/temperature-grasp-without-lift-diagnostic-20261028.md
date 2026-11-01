# Temperature-start recovery: grasp signals without lifting

The archived behavior is dominated by **sustained near-table grasp signals and coupled horizontal hand/object movement**. It is not predominantly a brief grasp followed by dropping, and the trace does not show an upward-moving TCP leaving the object behind. This is a descriptive diagnosis of recent training behavior, not a new policy evaluation or causal identification.

The frozen development gate remains **partial/inconclusive**. Lowering initial alpha recovered grasping, but not lifting or stacking:

| Policy/checkpoint | Grasp /128 | Lift /128 | Stack /128 |
|---|---:|---:|---:|
| Low-alpha, 51,200 transitions | 49 | 0 | 0 |
| Low-alpha, 102,400, primary | 64 | 1 | 0 |
| Low-alpha, 256,000 | 121 | 0 | 0 |
| Historical SAC, 256,000 | 128 | 121 | 0 |

Historical SAC took 1,259.63 trainer seconds to its 256k checkpoint; low-alpha used 192.38 checkpoint seconds /195.43 total child-process seconds. The historical contrast changes many recipe/compute factors and establishes an early-skill regression, not its cause. No historical critic/logstd reconstruction is possible from the retained exports.

## Provenance and alignment

Input is `runs/temperature-sufficiency-20261028/run/actor-step-256000-trajectory.npz`, SHA256 `fdd401f214a41f8f7dcd26df6c7d4797b0dd0e3be3ce7ab8fc6750cf123d6332`, with numeric state snapshot SHA256 `ed55d938dcf21a5e80a1268921303ebed23d67dbb556b761cdd736f761f4ae6f`. Both match the archived inventory. Their `actor-final` counterparts are byte-identical and were **not counted again**. The analysis covers 64 lanes ×50 control steps =3,200 already-recorded transitions, not 128 episodes or 256k independent samples.

Every lane truncates only at the last step; there are no termination flags. `obs[t+1] == next_obs[t]` exactly for all interior steps. The archived robot source records flags after `env.step` and stores its corrected terminal observation, so flags align with `next_obs`, not `obs`. These are complete aligned training episodes, but the actor can update during them; they are not frozen-policy rollouts.

The observation mapping was checked against installed pinned ManiSkill **3.0.0b21** source: `BaseEnv._get_obs_state_dict` emits agent before extra; `get_proprioception` emits qpos/qvel, with empty controller state in this mode; `flatten_state_dict` preserves insertion order; StackCube emits TCP, cube A/B and three relative positions. Zero-based half-open slices are:

| Fields | Indices |
|---|---|
| qpos / qvel | `[0:9]` / `[9:18]` |
| TCP pose / cube A pose / cube B pose | `[18:25]` / `[25:32]` / `[32:39]` |
| TCP→A / TCP→B / A→B position | `[39:42]` / `[42:45]` / `[45:48]` |

Pose position is the first three entries. Actor/link getters subtract GPU scene offsets when applicable, so the stored positions share scene-local coordinates. All three relative-position identities agree in pre/post observations: exact under float32 arithmetic, with maximum residual **1.49e−8m** when recomputed in float64. Task source matches frozen SHA `ff76ae60…`; base environment and both controller modules match previously downloaded commit `6226567…` byte-for-byte. The receipt hashes all 12 API/URDF sources used; no simulator was instantiated.

## Grasp duration and object motion

There were **61/64 ever-grasped**, **53/64 still flagged at the final step**, and **1,904/3,200 grasp frames**. Among ever-grasped episodes, median first grasp was control step **14** (range 6–27), and median longest continuous grasp run was **33** steps. There were 84 grasp segments and 31 internal true→false transitions, affecting 17 episodes; 14 episodes regrasped. Endpoint-held segments are censored by episode end, not counted as drops.

For a descriptive duration split, 56 episodes had a ≥10-step continuous grasp signal without a 3 cm rise; 5 had only shorter grasp signals; 3 never grasped. Ten steps is an analysis label, not a benchmark rule or proof of a secure grasp. The pinned Panda signal checks two finger contact forces/angles; raw forces were not saved, so load-bearing grasp strength cannot be inferred.

No lane reached the existing evaluator's **initial cube-center height +3 cm** lift threshold. Median per-episode maximum rise was **7.55 mm**, maximum **17.91 mm**. Median final horizontal cube displacement from reset was **18.04 cm** (mean 21.95 cm), so substantial movement occurred without lifting.

During the 1,820 frames continuing an existing grasp:

| Quantity | Mean | Median |
|---|---:|---:|
| Cube horizontal displacement/control step | 7.52mm | 5.33mm |
| TCP horizontal displacement/control step | 7.71mm | 5.62mm |
| Cube vertical displacement/control step | +0.208mm | +0.164mm |
| TCP vertical displacement/control step | +0.072mm | +0.103mm |
| TCP–cube distance | 10.99mm | 10.07mm |
| TCP/cube step-displacement cosine, moving pairs | 0.954 | 0.997 |

The median TCP/cube displacement difference was 0.62 mm per continuing-grasp step. Drop transitions instead had a median 6.16 mm mismatch and 29.33 mm separation. Thus some grip loss exists, but the dominant observed pattern is planar co-motion near table height. No wrench, penetration or causal contact diagnosis is available.

## Commands and checkpoint actor

The verified controller maps actions 0–6 to **seven joint-position deltas ±0.1 rad**, not Cartesian xyz. Action 7 is the mimic-finger position target, normalized to `[−.01,.04]` m; larger values request more opening. A large positive arm coordinate cannot be called an upward command.

During continuing grasp, actual joint 2 action averaged **+.975**, with 93.3% having absolute value≥.95. Its requested delta averaged +.0975 rad versus observed +.0117 rad; no recorded arm joint was within .01 rad of its URDF hard limit. This shows a command/tracking discrepancy, not whether contact, actuator dynamics or the learned command choice caused failure. Actual gripper action averaged +.567 (SD .631), corresponding to a mean mimic target 29.17 mm. This is a target, not measured width or grip force.

The final 256k actor was evaluated numerically on all 3,200 stored pre-step observations, with no sampling or fitting. Its deterministic outputs match its safe export exactly in this NumPy calculation; weights digest is `98c1be7b10ad8048a3cc56ad11307ee554d44d7b8920cff02563d40428796a0a`. During continuing-grasp observations its mean deterministic joint 2 action was +.984. Raw latent standard deviations averaged `[.226,.517,.167,.136,.377,.247,.430,.416]`; these are **counterfactual frozen-checkpoint parameters**, not the historical behavior parameters or post-tanh action standard deviations. Actual actions and behavior log probabilities are summarized separately. Checkpoint alpha was .004172. No Q-gradient was treated as truth.

The pinned reward assigns `(4+place_reward)/8` whenever the grasp flag is true, including near-table grasps; observed continuing-grasp reward averaged .538 versus .168 before first grasp. This provides a concrete incentive compatible with the pattern, but does not prove the actor exploited a local optimum or identify a remedy. The completed trace already distinguishes persistent planar contact from primarily transient dropping; another GPU job is not needed for that distinction. Any proposed intervention still needs a separate causal case.

## Reproduction and cost

New read-only script: `research/discovery/temperature_grasp_diagnostic.py`; full receipt: `research/discovery/temperature-grasp-diagnostic-20261028.json`, including all 64 episode summaries, phase geometry/action distributions, source/input hashes and cached aggregate references. Numeric NPZ loading used `allow_pickle=False`, archive/member bounds and inventory hashes; no pickle or archived candidate code execution occurred.

Analysis wall/process CPU time was **0.1392 /0.1280 seconds** after imports. Work: 3,200 frozen actor forward rows plus 3,200 safe-export parity rows, no optimizer calls, no new transitions, no scorer/confirmation queries, no cloud calls or trusted-file edits. Preliminary read-only array summaries preceded this recorded analysis; they did not fit or sample a policy. All original archives were preserved.
