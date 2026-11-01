# Short-history motion prediction did not meet the registered effect

The data-only CPU probe completed once and failed its material-effect gate in
both training seeds. It does not justify a recurrent learner or another GPU run.
It also does not establish that history is useless or explain PPO/SAC differences.

The target was next-frame cube-A displacement. Four held-out folds kept each
training-seed/environment-lane group together across every policy/checkpoint.
The registered primary subset used current geometry only: cube height>4cm and
TCP-to-cube distance<6cm. Errors below average within each eligible lane first,
then give each lane equal weight; lower is better.

| Fixed predictor | Seed1 primary error (mm) | Seed2 primary error (mm) |
|---|---:|---:|
| Current observation/action,128 Fourier features |12.213 |9.262 |
| Current plus previous displacement/action |9.662 |9.219 |
| Current plus matched-neighbor history control |12.522 |9.325 |
| Current observation/action,256 Fourier features |11.642 |8.975 |
| Current plus public-kinematic velocity features |11.686 |8.977 |
| Zero-motion descriptive baseline |16.105 |11.806 |
| Previous-displacement persistence baseline |11.393 |12.612 |

Seed1 contained17 eligible lanes/65 primary frames; seed2 contained50/287. Both
met the minimum16-lane eligibility rule, which is not a power guarantee. History
improved seed1 relative to the wider current predictor by17.0% and the kinematic
predictor by17.3%, below the registered20% requirement. On seed2 it slightly
underperformed both. The gate required at least20% and0.25mm improvement against
every fitted control in **each** seed; neither seed passed.

These seed-defined corpora have different checkpoint/collector composition:
61/65 seed1 primary frames came from coordinate exploration at102.4k, while
201/287 seed2 frames came from coordinate exploration at512k. This is not a
matched causal seed-replication comparison. The result only fails the registered
requirement for a large benefit in both fixed corpora.

The paired lane-resampling intervals for current256 error minus history error
were[0.451,3.657]mm and[-0.920,0.447]mm. These are conditional descriptive bootstrap
intervals for these fixed cross-fitted predictors, with overlapping training
folds and one random-feature draw. They are not exact unconditional95% coverage
or training-seed population uncertainty.

All-row frame-weighted errors did favor history:4.596mm versus6.043mm for
current256 and5.923mm for kinematic features. Simple previous-displacement
persistence already achieved4.731mm. This secondary result must not rescue the
failed primary or be represented as a control/learning improvement. Per-checkpoint
and all phase-bin/primary strata remain in the raw receipt, including sparse ones.

## Validity and cost

Fifteen existing50-frame training windows from five learners/two training seeds
provided15,360 subsampled transitions. No protected evaluation data, stage flag,
reward, oracle object velocity or newly collected simulator state entered fitting.
Within-episode observation continuity and relative-position layout checks were
exact. Whole seed/lane groups never crossed supervised train/test folds.

The kinematic comparator used measured robot qpos/qvel and public URDF geometry,
including a rigid-attachment cube-velocity proxy. This is explicit structural
assistance for a diagnostic predictor, not simulator truth or a learner change.
FK agreement was7.34micrometers/2.45microradians;64 directional Jacobian checks
had maximum arithmetic error2.98e-10. Counted15,488 analytic FK and15,360 Jacobian
pose evaluations. The assumption can fail during slip/release.

The neighbor-history control is label-free, nonbijective and transductive within
each held-out partition. It is not a conditional-randomization test. Wider random
features do not guarantee a stronger approximator under fixed ridge regularization;
model misspecification and sparse moving examples remain alternatives. A history
benefit could also simplify nonlinear current-state kinematics rather than reveal
irreducibly hidden information.

Actual child-process wall time2.754 seconds; internal diagnostic clock2.111 seconds
excludes imports/final serialization. Twenty closed-form ridge fits,230,400 fit
rows,76,800 prediction rows and61,440 neighbor queries; no gradient optimizer
calls. Zero new environment interactions, simulator queries, GPU seconds or
scoring queries. Original data-collection/training costs remain in historical
records; no new robot training run was counted. Scientific total remains72/40.

Source, input hashes, preregistered controls and the pre-fit kinematic amendment:
[plan](../research/discovery/history-motion-plan-20261029.json).
[Numeric result](../research/discovery/history-motion-prediction-20261029.json)
and [out-of-fold predictions](../research/discovery/history-motion-prediction-20261029.npz)
are preserved. No refit, threshold relaxation or model search was used after this
result. The initial GitHub review snapshot predates execution and remains unchanged.

Independent review recomputed predictor errors, cluster assignments, phase
summaries and bootstrap gates from saved predictions without refitting. Raw
geometry, continuity, persistence inputs and FK checks are supported by the
reviewed source/runtime gates; the predictions-only archive does not independently
reconstruct those predicates.
