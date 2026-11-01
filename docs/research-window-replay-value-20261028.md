# Split-BN mean constraint: actual-Critic identity and one fixed-policy fit

2026-10-28. Same replay-moment mechanism; CPU only. This follows the
[moment review](research-window-replay-review-20261028.md). No robot task,
actor learning, protected data or infrastructure access was involved.

## Preregistered claim and exact caveats

Let the final hidden normalizer independently center each half of a paired batch.
During BN warmup, for feature j and half h:

`z_hj = gamma_j * (x_hj - mean_hj) / sqrt(var_hj + epsilon) + beta_j`.

Therefore `mean_h(z_hj)=beta_j`, including constant features and any finite
epsilon. With an affine output head `Q_h=w^T z_h+b`, both halves satisfy

`mean Q(current) = mean Q(next) = w^T beta+b`.

This holds for arbitrary initialized or learned weights, scales and offsets,
provided the final normalization is immediately followed by an affine head.
Earlier nonlinear layers do not remove the constraint. Floating reductions
introduce rounding error. An activation **after** that final normalization would
invalidate the simple identity.

The identity concerns each head's **training-mode, warmup-BN** output. It does not
hold generally in running-stat inference mode or postwarm BRN. Nor does it imply
that the minimum of two heads has equal half means: disagreement can differ
between halves even when each head's means match. With one head, no entropy and
no terminal masks, a zero-mean Bellman residual would consequently require

`c = E[r] / (1-gamma)`.

That need not equal the true Q mean over off-policy replay. A small self-TD loss
can therefore accompany a large value-offset error. The robot learner's twin
minimum and entropy modify this fixed-point expression; do not import the
one-head numerical prediction unchanged.

## Fixed design, recorded before outcomes

The parent-specified scalar system was used exactly:

- `s'=.8s+.5a`, `r=-(s²+.1a²)`, fixed `pi(s)=-.6s`, discount .95.
- Shared fixed replay:16,384 one-step transitions, independent `s,a~U[-1,1]`;
  separate4,096 held-out transitions. Scalar state/action occupy the first of
 48 observation/eight action coordinates; the others are zero.
- Actual `Critic` from `research/discovery/replay_learner.py`, width64, one head.
  Three methods: ordinary target-network/no-BN, target-free joint moments,
  target-free split moments. All share each seed's linear initial weights and
  exact minibatch indices.
- Three seeds,1,000 updates each, batch256, Adam LR .001/betas(.5,.999), target
  EMA .005, original100,000-update BRN warmup. No entropy, actor or temperature
  update. No tuning, extension or retry was performed.

The fitting functions accept only observations, actions, rewards and successors.
They use ordinary Bellman targets. Analytic values are calculated exclusively by
the post-fit held-out measurement function:

`P = (1+.1*.36)/(1-.95*.25) = 1.358688524590164`,

`V_pi(s)=-P*s²`,

`Q_pi(s,a)=-(s²+.1a²)-.95*P*(.8s+.5a)²`.

No optimizer sees these analytic values. Because the dynamics and policy are
deterministic, the exact one-step target equals this Q, giving an unambiguous
held-out target-error measurement.

## Actual initialized Critic gate

The actual source's split mode gave current/next means .039104849/.039104827,
against final affine offset .039104819: maximum identity error **2.98e-8**.
The joint mode's half-mean difference was−.378893; it has no equal-half constraint.
For the split critic, inference-mode gap was−.001162 and artificially enabling
postwarm BRN gave−.011397, illustrating the identity's scope rather than claiming
those regimes were trained. The real toy fits stayed entirely in warmup BN.

## Results of the one fit

Held-out analytic means were **Q(current)=−.731125**, **Q(next policy)=−.388178**:
the true half-mean gap is **−.342947**. Separately centering both halves forces
that gap to zero in the warm training view.

| Seed | Method | True-Q RMSE, training view | Q mean bias | Actual bootstrap-target RMSE | Held-out self-TD MSE |
|---|---|---:|---:|---:|---:|
|1|Target network|.04462|−.03298|.03950|.000212|
|1|Joint moments|.10434|+.09744|.12762|.002304|
|1|Split moments|6.63251|−6.60889|6.62959|.009054|
|2|Target network|.12365|−.11867|.12031|.000087|
|2|Joint moments|.11644|−.00756|.07682|.003082|
|2|Split moments|6.72537|−6.66974|6.71076|.011213|
|3|Target network|.10432|−.09748|.10192|.000061|
|3|Joint moments|.24669|+.24124|.21941|.003208|
|3|Split moments|6.98762|−6.64778|7.00036|.018506|

Split-mode training mean gaps after fitting were0,4.77e-7 and0. Their predicted
current Q means were about−7.34,−7.40 and−7.38, near the population constraint
`E[r]/.05 = -(1.1/3)/.05 = -7.3333`, despite the true held-out mean near−.73.
Thus this test distinguishes true target error from a comparatively small
self-consistency residual. Joint mode avoided that gross error but **did not
outperform the competent target-network control**: target control had lower
Q/target RMSE in two seeds, joint in one. No joint-method superiority is claimed.

### Measurement issue preserved and excluded

The script also emits fields named `running_inference_*`. Review found that its
measurement clone first executes the held-out training-mode pass, advancing its
EMA once, then evaluates inference predictions using those slightly changed
statistics. These fields are **not untouched-trained-stat inference estimates**
and are excluded from every conclusion above. They did not affect learner
parameters, fitting batches, training-view Q/target errors or self-TD losses.
The original source/receipt are preserved; no rerun to improve an outcome was
made. A separately reviewed measurement-only correction can address this later.

## Interpretation and falsification

The algebraic prediction survived the actual-class check and three fixed toy
fits. This is a specific negative condition for **split final-layer centering**,
not a proof that normalized targets generally become accurate or that a robot
will improve. In this toy, a competent unnormalized target-network critic already
works; defeating split normalization alone would be a weak robotics claim.

For the ongoing robot screen, the appropriate interpretation remains fixed:

- Joint beating split supports avoiding this restriction in that recipe, while
  target-SAC remains the meaningful control for useful improvement.
- Joint not beating target-SAC provides no evidence of superior sample efficiency.
- Actor exploration, twin-head disagreement, entropy, replay coverage and physical
  contact dynamics are absent here. This test does not establish which of them
  dominates a robot result.
- A strong next-state/current-state distribution difference is necessary for this
  particular mean constraint to conflict with truth. If true means coincide,
  or the readout no longer follows a separately centered layer, the argument does
  not predict failure.

No new algorithm, reward change, teacher track or parameter search follows from
this CPU result.

## Receipts and accounting

- Source `research/discovery/replay_value_probe.py`, SHA256
  `6396c278ee8d0b5297736607cc52b7bb536777a5b7a94b8b66fde246e3374afc`.
- Actual owned critic source SHA256
  `009b4662a3b248e8e677905e2e7cafe2a69611a9259ab5fe91ec0cbe3b2efdfc`.
- Receipt `research/discovery/replay-value-probe-20261028.json`.
- **8.4065 seconds** measured script runtime;9,000 optimizer calls and2,304,000
  sampled fitting rows, shared16,384 unique synthetic training transitions and
 4,096 synthetic held-out transitions. Identity/held-out forward checks add
  computation, all included in that runtime. Robot transitions0, teacher rows0,
  actor updates0, GPU use0, cost$0. Source unchanged during execution.

The registered result is complete; these are initialization/minibatch seeds in a
fixed scalar dataset, not independent robotics training seeds.

## Reviewed measurement correction (same day)

One measurement-only repeat was authorized after the issue above was identified.
The original source and receipt remain in
`research/discovery/replay-value-original-v1-20261028/`; the corrected source is
`research/discovery/replay_value_probe.py`, SHA256
`d089353e33d8c9ee3f0df6691f19b459bb8dc55630443faf9c2ed93476f08987`.
Its receipt is
`research/discovery/replay-value-probe-measurement-fixed-20261028.json`.
The training-loop AST and hyperparameters are identical, and all unchanged fit
metrics exactly match the original receipt. The historical original parameter
digest was not saved, so bitwise parity with historical fitted parameters is
not claimed. This is not an independent replication or another tuned candidate.

The correction measures running-stat predictions and action derivatives from
an untouched fitted-state clone, and verifies that measurement preserves its
state. It does not advance normalization statistics on the held-out set.

| Method | Untouched running Q RMSE, seeds1/2/3 | Action-derivative RMSE, seeds1/2/3 |
|---|---|---|
| Target network | .0446 / .1236 / .1043 | .1256 / .1070 / .1145 |
| Joint moments | .1230 / .1096 / .2458 | .2528 / .2252 / .2509 |
| Split moments | 7.3081 / 8.0419 / 9.4205 | 1.7263 / 3.6678 / 9.0842 |

Joint moments avoid the split-mode calibration pathology, but the target-network
control has smaller action-derivative RMSE in every seed. Thus even improved
value calibration relative to split normalization does not establish better
policy-improvement gradients than the competent control. High derivative sign
agreement alone can hide badly wrong magnitudes.

The repeat took8.9862 CPU seconds. Both fits together consumed **17.3927 CPU
seconds,18,000 optimizer calls and4,608,000 sampled fitting rows**, reusing the
same16,384 synthetic training rows and4,096 held-out rows. These counts include
the measurement correction and must not be reported as only the first fit's
cost. Robot transitions, teacher rows and GPU use remain zero.
