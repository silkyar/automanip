# Three-arm episodic-risk continuation: tested source handoff

Status: CPU mathematics and marked-patch tests pass. **Not launched by this agent.**
The owner reviews and launches the isolated integration run, handles input bytes,
deadlines, cost accounting and protected scoring. This is an integration/early
quality pilot, not confirmation or a 99% reliability claim.

## Exact payloads

Public base: `research/discovery/upstream/precision-constant-base.py`, SHA256
`cd7ee5704baa451c0fb616140be2b8efc1830ae1dd3d355fc7d83e435b411082`.
Builder: `research/discovery/episodic_risk_precision_v1.py`.

| Arm | Self-contained file | SHA256 |
|---|---|---|
| Strong PPO control | `research/discovery/risk-pilot-v1/risk-ppo-control.py` | `5f802501efb026560774d795f16bacb0b8fc7767a17822f5c191b1431fc7988b` |
| Matched mean episode score | `research/discovery/risk-pilot-v1/risk-episode-mean.py` | `3c5fbf9334a16a1cdb584000ff74c49e05f645e9177488f079ca2e917142c2f0` |
| CVaR 10% episode score | `research/discovery/risk-pilot-v1/risk-episode-cvar10.py` | `ebf5469003849787e5d8469a48a76f94eb232f0554b8c8f0a07e0c5b85f3495c` |

Removing the three `DISCOVERY_RISK` marked blocks recovers the exact base bytes.
The Agent class, Gaussian sampling/density, critic targets, optimizer and precision
audit blocks are unchanged. There are no runtime imports of `research` or
`automanip`, no learned extra model, and no added parameters in agent serialization.
The owner's existing launcher must retain its usual internal-evaluation disabling,
trusted checkpoint staging, export, hard deadline and save-reserve handling for
these common-base bytes. The candidate patch does not replace that wrapper.

## Objective and diagnostics

For each complete 50-step ordinary training episode,
`G=sum_t gamma**t*reward_t`. The CVaR arm uses
`w=(G-q_alpha)*I[G<=q_alpha]/alpha`, alpha .1. The mean-score arm uses the same
formula at alpha 1, giving the conventional mean-return score in population.
Every step of one episode receives its same detached weight, divided by the
batch RMS; minibatch advantage centering is disabled for these two arms.
Ordinary PPO retains its original GAE and advantage normalization. Critic targets
remain original GAE returns in all arms. The task reward itself is unchanged.

At the behavior policy, the episodic score identity is exact in population;
averaging over time introduces a common factor 1/50. Empirical VaR, RMS scaling,
per-step PPO ratio clipping and repeated epochs are approximations. This is not
an exact unbiased finite-batch optimizer of static episodic CVaR, nor recursive
statewise CVaR. No ordinary state baseline is subtracted inside a selected tail.

Each rollout rejects partial/cross-episode records and requires the pinned public
wrapper's pre-reset `final_info.episode.success_at_end` boolean vector. Success
flags are used **only** for training diagnostics; changing them leaves actor
weights unchanged in a regression test. Std and optimizer audit tags remain:

```text
audit/optimizer_steps
audit/training_entropy_coef
audit/logstd_dim_0 ... audit/logstd_dim_7
```

Additional `audit/risk/*` tags and `AUTOMANIP_RISK_DIAGNOSTIC` JSON lines contain
return quantile, strict/inclusive tail sizes, training final-success count,
tail failure rate, failure capture and enrichment. No failures gives null
enrichment/capture, not a misleading zero. Return ties have zero score residual;
strict and inclusive membership are both reported. All arms log the proposed 10%
tail so objective relevance can be compared without using success as a reward.

If low-return tail membership does not enrich final training failures, report
that the dense-return risk objective may miss the late stability problem. No
reward addition, discount sweep or threshold adjustment is authorized by this
pilot. A final development result is interpreted alongside this mechanism check.

## Owner-proposed bounded continuation

All three arms start from the same OUR independently confirmed reference:

```text
native input SHA256: 307a2bc5996d5def9f696ab35b27a9cf96a5cf7401c678ecc981b78170d1084b
prefix transitions: 24,985,600
prefix trainer time: 1,088.77 seconds
optimizer state: restarted in every arm
```

These values were provided by the owner; this agent did not access protected
checkpoint paths. Stage only the reviewed ordinary native Agent state_dict in
the trainer mount. No teacher data, demonstrations, external pretrained weights,
planning queries or protected evaluation inputs are needed.

The owner proposes 2,048,000 **new** transitions per arm under a 300-second lease
with a 20-second save reserve. Per-policy accounting must therefore report
27,033,600 prefix-plus-new transitions if the requested increment completes, and
1,088.77 plus actual new trainer seconds. The pilot's additional training ceiling
is 6,144,000 transitions and 900 resource-seconds across all arms, before the
separately counted evaluation/startup work. This exceeds a fresh 25M comparison
when its prefix is included and must be labeled a continuation track.

Common CLI, after the owner applies the existing isolated launcher wrapper:

```sh
python /workspace/candidate/risk-episode-cvar10.py \
  --exp-name risk-episode-cvar10 --seed 1 \
  --env-id StackCube-v1 --control-mode pd_joint_delta_pos \
  --checkpoint /workspace/reference/confirmed-seed1.pt \
  --total-timesteps 2048000 --num-envs 1024 --num-steps 50 \
  --num-minibatches 32 --update-epochs 8 \
  --gamma 0.9 --gae-lambda 0.9 --ent-coef 0.003 \
  --learning-rate 0.0003 --target-kl 0.1 \
  --no-partial-reset --no-capture-video --no-track
```

Use the respective file/experiment name for the other two arms; all scientific
CLI settings and input bytes match. The trusted wrapper supplies the absolute
`AUTOMANIP_TRAINING_DEADLINE` once and an independent hard stop; a queued process
must not receive a renewed lease. The wrapper's initial/final exports must retain
ordinary actor/critic state_dict keys, and the trusted evaluator receives only
its approved safe deterministic actor artifact. No evaluator edits are needed.

Protected scoring: one final 128-episode development evaluation per arm, handled
by the owner; no confirmation or final holdout queries. This cannot certify 99%
success or establish across-seed reliability. A positive result requires later
independent matched full-training seeds and separately budgeted confirmation.

Proposed **additional** hard spending cap: $1 including all three trainings,
evaluation, startup and overhead. Actual dollars and the owner's current provider
rate must be recorded before/after launch; no rate is invented here. The owner
must enforce that cap or defer. Previously spent reference compute is reported
separately, not silently treated as free data. No retry is included.

## CPU evidence and reproduction

`research/discovery/risk-pilot-v1/cpu-receipt.json` records the exact payload hashes
and nine enumerated stochastic score-gradient checks. Maximum absolute error
versus exact finite differences: 1.24e-11. Six focused tests additionally cover
behavior-policy PPO/RMS/horizon scaling, return discounting, episode boundaries,
diagnostic-only success labels, ties/null denominators, byte-removable patches,
unchanged Agent/critic targets and preserved audit tags.

```sh
.venv311/bin/python -m unittest discover -s tests -p 'test_discovery_episodic_risk_v1.py' -v
.venv311/bin/python -m research.discovery.episodic_risk_precision_v1 \
  --base research/discovery/upstream/precision-constant-base.py \
  --method risk-episode-cvar10 --output /tmp/new-risk-cvar10-payload.py
```

The builder refuses a changed base, an unknown method or output overwrite.
Source/CPU tests are ready for owner review; no robot policy performance has yet
been measured for this mechanism.
