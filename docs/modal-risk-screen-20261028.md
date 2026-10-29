# Modal risk continuation pilot operations

The three-arm pilot used the reviewed standalone programs in
`research/discovery/risk-pilot-v1/` and frozen scientific plan
`docs/stackcube-risk-pilot-plan-20261027.md` (SHA256
`229ac10a532fd0b95529d6f4f67f398b5468893bef04d540aaf1b95ae42b6f86`).
Root reviewed the complete wrapper/contracts and learning patches, passed six
learning and five wrapper CPU tests, and verified reference actor conversion.
After automatic review initially rejected the upload, the user explicitly approved
it. Root submitted the exact action through ordinary review; no duplicate launch
was submitted by the integration worker.

App: `ap-GrmuozFJJluNhZsii6fw4D`. Coordinator:
`fc-01M601WZMK86NWZ9CXW438FPQP`. The admitted plan SHA256 is
`7f1ba187b3d8816dd2a8cb227a46814037f30262c3ea058c03032ab5c7272fbe`.
All three trainers used actual NVIDIA L40S hardware. Sequential dispatch ensured
that each original 300-second lease began only after the prior trainer ended.
Every trainer verified the same 1,168,418-byte own-project checkpoint SHA256
`307a2bc5996d5def9f696ab35b27a9cf96a5cf7401c678ecc981b78170d1084b`.
No confirmation or private holdout was queried.

This was continuation from the project's confirmed reference, with optimizer and
random stream restarted. Every result preserves the prefix of 24,985,600 training
transitions and 1,088.7667 trainer seconds. It is not a fresh-from-scratch result.
Each arm collected exactly 2,048,000 new transitions, so its prefix-plus-new count
is 27,033,600. The original prefix is shared data, not three new training runs.

| Arm | Admission UTC | New trainer seconds | Actual optimizer steps | Final development |
|---|---|---:|---:|---:|
| Ordinary PPO | 01:27:32.538525 | 85.566 | 1,889 | 123/128 |
| Episode mean | 01:29:06.989488 | 146.272 | 9,289 | 104/128 |
| Episode CVaR10 | 01:31:42.537736 | 96.085 | 256 | 91/128 |

The learning-code diagnostics and exact audit traces are preserved for scientific
interpretation in the separate pilot report. Optional tail-failure diagnostics
were permitted to be absent/null when there were no training failures. Training
success flags were diagnostic labels only; the actor objective used rewards.

All 43 output files, 24,855,671 bytes, are archived and hash verified under
`runs/stackcube-risk-continuation-20261028/`, including sources, reference copies,
native checkpoints/events/logs/scalars, safe actors and full frozen reports.
Summary SHA256:
`84abb489eaa6f0e097637ffcafb50c408edcfdfef7cf96a8d77cf14d31d5c442`.
The operator archived native files as bytes without local deserialization.

The app automatically stopped at 01:34:46 UTC on 2026-10-28. Exact lifecycle and
zero-task readback passed; no cleanup mutation was needed and output Volumes and
once-only claims remain intact. Reported provider cost was $0.29366872 at close-out,
subject to meter lag. The one shared `.runtime/modal-tonight-budget-ledger.json`
includes all AutoManip Modal setup/failures/training/evaluations and outstanding
reservations under the user's $100 total cap. No new Runpod spend was initiated.
