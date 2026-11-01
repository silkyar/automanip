# Fresh-seed full-task learning curves under a ten-minute ceiling

Registered before training, following independent review of the scientific case.
The earlier joint-exploration primary failed; its coordinate control unexpectedly
learned34/128 lifts versus the cached ordinary low-alpha reference's1/128 at
102.4k. Both scored zero stacks. This is a new bounded study, not a promotion or
extension of either old run. No old policy/replay/optimizer is resumed.

## Methods and competing hypotheses

Two fresh **training seed2** learners: ordinary low-alpha SAC versus the unchanged
coordinatewise epsilon.1 behavior collector. Same random initial actor, online
critics, target critics, alpha and optimizer; same task/reward/horizon48/8 interface,
64envs, replay/update/minibatch budgets, gamma.99 and low-alpha.005 initialization.
The baseline uses the collector's proven epsilon0 identity path so proposal/RNG
and small collector overhead remain matched. Actor/critic/temperature/target
learning code is unchanged. No teacher, diagnostic state/action, analytic controller
or task-stage oracle enters either learner.

H_effect: the coordinate collector's seed1 lift signal survives a new seed and
improves full stacking acquisition. H_stage_only/null: it changes early-stage
behavior but does not improve task completion under this budget. Similar successful
curves favor the simpler ordinary SAC. Neither one fresh seed nor a historical
PPO comparison establishes algorithm superiority or a general10x mechanism.

## Frozen endpoints and decisions

Each run has **dual ceilings:1,024,000 transitions and600 child-process seconds**,
including imports and final export. No lease extension, resumed training, endpoint
replacement or automatic new seed. Missing512k is censored, not evidence at512k.

Primary collector comparison: both exact512,000-step frozen development policies.
A promising directional effect requires coordinate >=64/128 final stacks AND
at least20 stacks above baseline at this checkpoint. Coordinate <=8/128 weakens
full-task sufficiency at512k; intermediate results are partial. Missing either
primary leaves this comparison missing. A clear regression or coordinate<=baseline
weakens the collector's benefit. No later endpoint rescues a failed primary.

Separately report the research objective as **first observed** >=103/128 (80%) and
>=116/128 (90%) development crossings at the declared checkpoints. Give observed
checkpoint brackets, no interpolation or assumption of monotonicity. A crossing
on this one-seed adaptive development set is not confirmed success or proof of
tenfold efficiency. The historical strong PPO curve is context, with task,
hardware and timing definitions disclosed; it is not a newly matched algorithm
comparison or the sole denominator after a stronger off-policy baseline exists.

Frozen development points per policy:102.4k (secondary early-stage replication),
256k,512k, and the terminal safe actor if its digest is distinct. If1.024M is
reached, terminal is that exact-step checkpoint; if time censors earlier, record
actual steps/time and score terminal only as the preregistered terminal endpoint.
Do not call terminal policies equal-600s policies when a step ceiling ends a run
early. Deduplicate unique hashes and reuse durable reports. At most8 new128-episode
development queries, zero confirmation/private-final-holdout queries.

Record complete success/stage curves, training/initialization/export times,
optimizer and collector work, masks, actual environment transitions and scoring
compute separately. All failed/censored attempts remain archived. Save actor,
online critics, **target critics**, alpha and RNG at the same checkpoint instant
for later lag diagnostics. Saving targets is common serialization work, not a
learning change or reconstruction of historical optimizer/replay updates.

## Execution and admission

Bind a newly computed trusted **seed2 initial safe-actor hash**; never reuse seed1's
hash. CPU-check epsilon0 identity, initializer/model/target equality and serialization
nonmutation/RNG preservation. Reports must use actual collector.epsilon (0 or.1),
not an inherited hardcoded.1. Review exact scientific/source/config/artifact pins
independently before admission.

TwoL40S workers: child<=600s, function<=660s plus30s startup, original worker
lease645s and no refresh. Trusted CPU supervisor<=900s only spawns/waits and
records scalar worker outcomes/call IDs; it does **not** parse candidate-volume
files or invoke scoring. This avoids the observed in-container Volume byte-padding
failure. External bounded reads hash-archive and validate all artifacts first;
only then invoke the unchanged once-only frozen evaluator from the local trusted
handoff. Candidate workers have no evaluator/private seeds/credentials/network.

One shared **$3 total campaign reservation**, including2worker/startup envelopes,
supervision, at most8 frozen120s/30startup evaluator envelopes and handoff/setup,
within the existing **$100 total session Modal cap**. No new Runpod spending.
Exact root admission/reservation and code review are required. Verify both apps
STOPPED/zero tasks and archive every output before closing the cap. No automatic
promotion, confirmation, replication or follow-up training from this study.
