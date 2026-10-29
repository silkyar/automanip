# Minimal StackCube publication repository

Prepared after the approved first-update probe completed and was archived.
The main AutoManip sample-efficiency research remains unresolved; no new GPU
training was started for this extraction.

Standalone local Git repository:
[`publication/stackcube-ppo`](../publication/stackcube-ppo/README.md).
It is independent of the parent Git history and imports nothing from AutoManip.
A portable Git bundle is at
[`publication/stackcube-ppo.bundle`](../publication/stackcube-ppo.bundle).

Contents: byte-identical official ManiSkill PPO, a small stdlib launcher,
official/improved JSON recipes, three-line improvement patch, exact aggregate
three-seed results, source/version digests, Apache attribution, dependency locks
and a compact Dockerfile. No cloud orchestration, research agent, protected
benchmark episodes, individual scorer outputs, credentials or checkpoints.

The direct headline is **32.23% → 94.08% independent confirmation success**
at 24,985,600 transitions per seed, across three training seeds on L4. The
remembered34% starting point is not the three-seed confirmation mean.
An intermediate discount/entropy recipe reached82.10%.

The learning-recipe change is exactly:

```diff
-  "gamma": 0.8,
-  "ent_coef": 0.0,
-  "partial_reset": true
+  "gamma": 0.9,
+  "ent_coef": 0.003,
+  "partial_reset": false
```

Git tags `official-baseline` and `improved-recipe` differ only in those three
recipe values. Shared update epochs, simulator transitions, state/action schema,
network, task and reward are preserved. The new Git history is explicitly a
retrospective explanation, not the original chronological research history.

Validation: both dry-run commands passed without Torch/simulator imports;
the vendored trainer matches the original source byte for byte; reported means
match the retained ledger; CSV digest verified; patch passes reverse-apply
validation; scanned the allowlisted files for credential/config/private-path
markers. Existing root frozen evaluator/protocol files were not modified.
Docker build, GPU training and scoring have not been rerun in the extracted
repository. Historical independent scoring remains evidence from the parent;
the minimal repo does not claim to reproduce the protected evaluator boundary.

This creates a local publication-ready repository and bundle. No GitHub remote
was created, and nothing was pushed publicly. Publishing destination/visibility
can be chosen separately without changing the documented experiment.

The requested progress figures and final25M-policy illustrative video were
subsequently added in publication commit`f02967fa4d24df1f5aba3b541273dae300b7b83d`.
The primary chart fixes24,985,600 transitions; the separate22-minute graph uses
only eligible recorded checkpoints and discloses unequal step counts. The
publication excludes50M quality runs from both comparisons. Source aggregates,
plot scripts, PNG/SVG/PDF exports and per-file hashes are included.

The video was rendered on L40S from the existing full-horizon seed1 policy,
trained24,985,600 transitions in1,088.77seconds on L4. The renderer used400
public transitions, no new training or protected evaluation. It selected the
first terminal success among8 declared public seeds; all8 happened to succeed.
This is an illustration, not a new success-rate estimate. The saved MP4 decodes
to51 frames at512×512,20fps; the final stacked/released frame was inspected.
Its receipt and public selection rule are retained in the publication assets.

The updated bundle was independently cloned and verified, with clean worktree
and unchanged three-line tagged diff. Bundle SHA256:
`7fbf02d6f41543d698a55b446a6f7726aff6b6a6b1e1969d0c0674420333621c`.
Video SHA256:
`6f5f4d2efbc741deafeb53b530b7b4ddb54c4aaf17b8062eb7c6c56b90de0d1d`.
No GitHub remote or public push was added.
