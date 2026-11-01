# Fixed demonstration-assisted AMP pilot

This one-seed diagnostic compares BC-only, BC+PPO and BC+PPO+AMP. It is a
prior-assisted track; teacher training, demonstration generation and behavior
cloning are additional experience/compute, not free initialization or evidence of
zero-demo sample efficiency. Initial cap $3 within the one shared $100 Modal cap.
No paid launch occurs until the complete adapter and this plan receive root review.

Use the project's own constant PPO teacher, safe actor SHA256
`ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`,
with its original seed-1 prefix 49,971,200 transitions /2,295.408300588 trainer
seconds. Exact per-teacher dollar allocation is unavailable inside its three-arm
app and remains null, not zero. That app's financial spend is already included
once in the session ledger; teacher training is not repeated.

Collect fresh public TRAINING resets **90,280,000 through90,280,007**, eight
batches of64 complete 50-step episodes, at most512 episodes /25,600 transitions.
The integer seed resets each vector batch; episode order preserves vector index.
These seeds were fixed before launch without reading benchmark episodes/seeds.
Preserve complete failures and any incomplete output. A failed collection aborts.
A budget stop before another full batch may use its complete episodes if at least
four successes remain; no refill or adaptive extra collection. Production requires
exact pinned Linux/CUDA packages and `physx_cuda`; all MacSAPIEN/CPU smoke data and
BC reports are ineligible. The collector has a120-second original dispatch lease.

Fit one random-initialized BC actor, student seed1, 5,000 Adam updates maximum,
1,024 examples/update, 120-second original lease. Only successful public training
demonstration episodes are selected. The source's fixed digest-based whole-episode
split is used; demo action-validation MSE is not protected task scoring. Preserve
random and final actors plus all BC/update/minibatch/source/data accounting.

Score the BC actor once on frozen128-episode development. Its report is accessible
only to the trusted coordinator. The recipe and budgets below do not adapt to this
score. Both matched students then receive the exact same safe BC actor, BC report
and dataset digest, never teacher weights. Their critic and learned action-logstd
initialization remain the same fresh seed1 PPO initialization. No native teacher
checkpoint loading is permitted.

Run the two reviewed standalone payloads `assist-bc-ppo` and
`assist-bc-ppo-amp` from `research/discovery/amp-prior-assisted-v1/`. Both keep the
public StackCube state48/action8/controller/task/reward interface, full50-step
horizon, 1,024 envs, gamma.9, entropy.003, LR.0003, 8epochs and32minibatches.
Each gets at most **5,120,000 new RL transitions or600 seconds**, whichever comes
first, with a20-second save reserve and independent external hard stop. Queue and
startup consume each original lease; preemption/retries never refresh it. Two
student trainers may run in parallel. Score each final actor once on frozen
128-development episodes. Max three unique development evaluations including
BC-only; identical actor hashes reuse the same protected report. No confirmation,
private holdout, extra seeds, automatic retry or extra GPU smoke is admitted.

AMP adds .25×bounded adversarial style reward to unchanged task reward, with32
fixed discriminator updates per rollout using successful demonstration transitions
and actual pre-reset learner successors. Preserve task/style logs separately,
exact PPO/discriminator updates and sampled rows, replay rows and wall components.
The full state48→successor48 discriminator is task-specific. Local controls showed
high style for out-of-distribution random failures and a small temporal-shift gap;
it is not a success classifier and useful temporal modeling is unproven.

Outcome rule fixed before results: no automatic expansion or confirmation.
Compare final successes at matching new-transition count only if both students
complete5.12M; otherwise report censoring, actual steps/time and no sample-efficiency
gain claim. AMP merits a separately planned follow-up only if it gains at least
3 paired development successes over BC+PPO and does not regress below BC-only.
A one-seed pilot cannot establish reliable near100% success. Eventual promotion
still requires fresh training seeds and independent confirmation. Do not choose
an intermediate checkpoint or tune from the BC-only/final scores.

Collector/BC/students have separate outputs; published dataset and BC actor/report
subpaths are readonly student inputs. Teacher numeric actor goes to collector
only. No candidate receives evaluator source/results, benchmarks or credentials.
Trusted scoring sees only validated numeric actor manifest/weights, never NPZ or
candidate Python. Preserve teacher prefix, all generation/BC/RL/discriminator and
built-in/frozen evaluation accounting separately. Shared assistance costs count
once per campaign and are disclosed as prefixes for each assisted policy.

Resource bound at existing documented rates:2,250 GPU-container seconds including
wrapper/startup/evaluation ceilings, two CPU conversions and3,600 seconds of
coordination; $1.4314206 subtotal and $1.5685794 reserve under$3. Provider metering,
setup/I/O and lag remain visible. The shared session ledger must contain one $3
reservation before launch. Exact sources/commands/seeds/mounts and this document's
SHA256 are bound by `deploy/modal/amp-pilot-plan.json` and its launch receipt.
