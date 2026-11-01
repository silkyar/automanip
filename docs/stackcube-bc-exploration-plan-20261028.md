# Fixed BC exploration pilot

The target is reducing student RL experience toward the independently confirmed
roughly96% reference quality. This diagnostic is a teacher-assisted track, not a
new from-scratch result or a claimed sample-efficiency gain. AMP is deferred;
its reviewed adapter, disabled plan and source remain untouched. Neither arm
uses a discriminator or reward shaping. This pilot reuses the existing $3
reservation inside the shared $100 Modal cap; root renames that reservation only
when separately admitting this replacement. No new allowance is created.

Reuse the reviewed demonstration/BC flow: own-project 50M constant PPO teacher
safe hash `ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`,
original seed1 prefix49,971,200 transitions /2,295.408300588 trainer seconds.
Exact per-teacher dollars are unavailable inside the original three-arm app,
not assumed zero. Its financial spend is already counted once; training is not
repeated. Generate at most512 complete50-step episodes in64-env batches, using
fresh public TRAINING reset seeds **90,281,000 through90,281,007**, frozen before
collection without consulting benchmark episodes or seeds. Preserve failures,
resets and partial outputs. Collection has one original120s dispatch lease;
failure aborts, budget censoring may use complete batches only if at least four
successes remain. No refill or CPU-data substitution.

Fit one random seed1 BC actor, at most5,000 Adam updates ×1,024 rows and one120s
lease, selecting successful public training episodes with the existing fixed
whole-episode split. Preserve all generation, BC, train/validation-index and
teacher-prefix accounting. Score BC-only once on frozen128-episode development;
its report is never a student input and does not change this fixed recipe.

Two fresh matched students use exactly the same safe BC actor/report/dataset,
mean/critic initialization and seed1 PPO settings. Both retain the frozen task,
state48/action8/controller, full50-step episodes,1,024 environments, gamma.9,
entropy.003, LR.0003,8epochs and32minibatches. Source identities:

- Standard `assist-bc-ppo`: `7a4de8deecf0adc15d9f8beaef1bf6b9874ca9eb7b0122efbab5d9b1bc80d427`.
- `assist-bc-ppo-residual-std` v2: `c5ca37a6ff22ddd2e89288f5697e3041507e834adf8f8731a49a17d1423fd200`.

The control keeps the reviewed constructor's initial logstd−.5 in all eight
coordinates. The v2 arm computes raw-mean action residual RMS using **BC training
episodes only**, clips each coordinate to[.05,exp(−.5)], and initializes only the
eight existing Gaussian standard deviations before Adam creation. Means,
critic, architecture, RNG and reward stay unchanged; std remains trainable under
ordinary PPO. This is an exploration heuristic, not calibrated posterior
uncertainty. Lower noise can prevent useful correction discovery; CPU examples
are evidence to keep the experiment small, not production constants.

Require the exact calibration receipt/schema, dataset/BC actor/patch/base hashes,
training indices and excluded validation indices equal to the shared BC report,
residual_forward_rows=50×training_episode_count, eight finite RMS/original/initial
std values, clamping consistency and all18 step-zero scalar traces. Calibration
adds zero simulator transitions/optimizer updates. Preserve its forward rows and
wall seconds; the full worker lease includes this setup even though upstream
SPS starts later. Trusted code checks receipts and hashes, never decompresses
candidate NPZ or executes candidate calibration code. Source-level CPU tests cover
the residual calculation and preservation of mean/critic/schema/RNG.

Each student gets **5,120,000 new RL transitions or600 original dispatch seconds**,
whichever comes first, with20s save reserve and independent hard stop. Queue/setup
consume that same lease; no retry/preemption refresh. The two may run in parallel.
Final actor only, one frozen128-development query per arm; at most three unique
queries including BC-only. No confirmation, private holdout, extra seeds or smoke.
Inputs are readonly dataset-only and final-BC-actor/report subpaths; teacher weights
reach collector only. Production rejects all MacSAPIEN/CPU smoke artifacts.

Outcome rule before results: no automatic expansion. Compare final counts at
matching5.12M new transitions if both finish; otherwise report censoring/actual
steps/time without a sample-efficiency gain claim. Residual initialization merits
a separately planned follow-up only at least3 paired development successes above
standard BC+PPO and no regression below BC-only. Do not cherry-pick checkpoints.
Reliable improvement still needs fresh student/teacher training seeds and
independent confirmation. Report all teacher/demo/BC/RL/calibration/built-in and
frozen-evaluation costs separately; shared assistance counts once per campaign.

Conservative unchanged wrapper envelope:2,250 GPU-container seconds plus two CPU
conversions and3,600s coordinator, $1.4314206 subtotal /$1.5685794 reserve under$3.
Actual meter, setup/I/O and lag remain visible. Exact sources, commands, seed range,
readonly mounts and this document SHA256 are bound by the separate default-denied
`deploy/modal/bc-exploration-plan.json`. Root reviews/admission precede any upload
or paid call. No source/evaluator changes or production GPU tests are implied.
