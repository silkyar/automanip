# AMP pilot readiness — CPU review only

**Default denied:** `launch_enabled=false`, `paid_launch_authorized=false`.
Proposed cap **$3** within the shared $100 Modal ledger. This note creates no
resource/reservation and authorizes no upload. Standing approval remains pending.

Reviewed `amp_collect_v1.py`, `amp_bc_v1.py`, `amp_v1.py` and both standalone
`amp-prior-assisted-v1/` payloads. All `source-receipt.json` hashes matched; payloads
compiled; **nine CPU tests passed in `.venv311`**, including production rejection
of CPU smoke data, true terminal successors, episode splits, safe export and
unchanged PPO schema. No research source or protected episode output was changed/
read. Local MacSAPIEN/synthetic checks establish dataflow only; no production GPU
or closed-loop sample-efficiency result exists.

## Small fixed pilot

| Stage | Proposed bound | Preserved output / accounting |
|---|---|---|
| Fresh GPU teacher collection | 512 complete 50-step episodes, 64 envs; 120s absolute lease | 25,600 teacher transitions maximum; all successes/failures, partial batches, resets, seeds, source/data/teacher hashes |
| One random student BC fit | Seed 1; 5,000 updates × 1,024 rows; separate 120s lease | Initial/final safe actors, BC report, demonstration rows and whole-episode train/validation split |
| BC-only frozen score | One 128-episode development query | Report remains trusted; cannot become a student input or tune this recipe |
| Matched BC+PPO / BC+PPO+AMP | Shared exact BC actor/report/dataset; seed 1; proposed 5.12M new steps and 600s original lease each | Same fresh critic/logstd initialization; actual/censored rollouts, PPO/discriminator work and final safe actors |
| Final frozen scores | One 128-episode development query per student | Three queries total including BC-only; no confirmation/retries/extra seeds |

Propose the own-project 50M constant teacher, safe hash
`ebe92bbc1ce5dec3f8a271792dd4befb81296b36a5f131345c84e8715a39ea16`.
Disclose its **49,971,200-transition /2,295.408300588-second training prefix**;
teacher dollar attribution is known or explicitly unavailable, never assumed free.
Predeclare fresh public training reset seeds without consulting benchmark episodes.
Failed collection aborts; a budget stop may use complete batches only if at least
four successes remain for the fixed split. No refill or CPU-data substitution.

BC receives state/action examples, never teacher weights. Successful training
trajectories are explicitly selected; raw failures remain archived. Demo holdout
MSE is not robot scoring. AMP adds bounded style reward at weight .25 and 32
discriminator updates per rollout. Its full state48→true-successor-state48 prior
is task-specific; useful temporal modeling remains unproven. Preserve separate
teacher-prefix, demo-generation, BC, RL, discriminator, built-in-evaluation and
frozen-evaluation steps/time/costs. Count shared demo/BC costs once per campaign,
and disclose their assistance for every policy; this is not zero-demo from scratch.

## Minimal integration / gates

Reuse existing risk/precision isolation, once-keys, dispatch-time deadlines,
20s save reserve, attempt markers, inventories, restricted conversion, frozen
scoring, archives and exact-app cleanup. Add collector/BC stages and readonly
mounts only. Collector/BC need an explicit three-file package allowlist for
`research.discovery.amp_v1`; students are already standalone. Source identities:

- BC+PPO: `7a4de8deecf0adc15d9f8beaef1bf6b9874ca9eb7b0122efbab5d9b1bc80d427`
- BC+PPO+AMP: `c2af05677a3a3652b0bd6b27f482a4946a51cad72b0ce0106e6b2b7b90f457ad`

Only collector mounts teacher actor JSON. BC mounts the dataset. Students mount
readonly dataset, shared BC actor JSON and BC report through `--assist-dataset`,
`--assist-bc-actor`, `--assist-bc-report`; **no `--checkpoint` or teacher weights**.
Each stage owns separate output storage. No trainer receives evaluator/results,
benchmark data or credentials. The roughly 5.94 MB raw dataset needs Volumes and
small receipts, not large restricted RPC returns. Bind lengths/digests and archive
limits. NPZ uses `allow_pickle=False`; no dataset or candidate code enters trusted
scoring, only validated numeric actor artifacts.

Production commands must forbid `--local-cpu-dataflow-smoke`. Default loaders
reject the MacSAPIEN backend/version; GPU BC rejects its CPU override and student
initialization rejects noneligible BC reports. Additionally bind real GPU collector
receipt, exact Linux/CUDA pins, `physx_cuda`, public-training provenance, source
hashes and dataset inventory; editable metadata alone is insufficient. Verify
complete/partial transition and reset accounting. Current CPU data/BC artifacts
are ineligible production inputs.

## Budget / blockers

Using the existing adapters' documented rates: collector+BC **2×(180+30)s**,
students **2×(660+30)s**, scorers **3×(120+30)s** = **2,250 GPU-container seconds**.
Including two CPU conversions and 3,600s coordinator ceiling gives **$1.4314206**,
leaving **$1.5685794** within $3 for setup/I/O/interruptions. Report actual meter/lag.
The 01:36 ledger had $7.72405866 completed plus $4.50 blocked replication reserved;
adding this unapproved proposal would total $15.22405866, below $100. Refresh that
single ledger before admission; no independent allowance is created here.

Remaining blockers: standing upload approval; frozen teacher/seed/transition and
outcome rules before results; exact stage/helper/command/mount admission; bounded
artifact/accounting wrapper review; and separate root paid approval. No adapter
was implemented or cloud action performed for this review.
