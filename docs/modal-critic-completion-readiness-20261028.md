# Critic completion pilot — local readiness outline

**Default denied:** no source registry, paid admission, upload or cloud call is
created by this note. Root must approve the mechanism, frozen source handoff,
CPU invariants and outcome rule before a separately admitted $3 pilot. Existing
BC exploration and deferred AMP files remain unchanged.

## Two-run envelope

Reuse the reviewed CAPG/precision two-source workflow directly, without dataset,
BC or teacher stages. Proposed training seed1 in both runs, random initialization:
strong ordinary PPO control versus critic continuation after actor-KL stopping.
Same frozen StackCube state48/action8/controller, full50-step episodes,1,024 envs,
gamma.9, entropy.003, LR.0003,8epochs and32minibatches. No checkpoint input,
demonstrations, reward shaping, new architecture or new simulation interface.

Each run requests **10,240,000 transitions** (200×51,200 exactly), with a **600s
original lease**,20s save reserve,660s wrapper and30s startup allowance. Two dedicated
L40S trainers may run concurrently. Queue/setup consume the original dispatch
lease; attempt markers and campaign/job claims prevent fresh retries.

Frozen development scoring: **5,120,000 transitions (`ckpt_101.pt`) and final
actual transitions**, maximum four128-episode queries across the pair. Reuse the
existing checkpoint selector with only `{101:5_120_000}` and a10.24M ceiling;
merge final/cadence coincidence, reject future/missing expected checkpoints and
record censoring. Final-only points below5.12M remain censored observations, not
invented policies. Identical weight hashes reuse one frozen result.

Preserve native checkpoints/events/logs/scalars and safe actors. Trusted CPU
coordination, restricted weights-only conversion, the existing frozen evaluator,
paired development seeds and safe artifact schema stay unchanged. Trainers see
only their own output Volume and the previously validated readonly PhysX cache;
no evaluator/results/benchmark mounts or credentials.

## Source/counter handoff required

The candidate's upcoming design continues critic-only updates through remaining
minibatches of the **same configured eight epochs** once the actor hits target KL.
This differs from the older local prototype's extra-one-value-epoch helper; do
not admit that older source by accident. Control retains ordinary break behavior.
No candidate payload hashes are frozen in this outline yet.

Planned `AUTOMANIP_KL_VALUE_UPDATES`: one JSON row per completed rollout, with
measured `joint_actor_critic_optimizer_calls`, `actor_optimizer_calls`,
`critic_only_optimizer_calls`, `critic_optimizer_calls`, `optimizer_calls_total`,
plus cumulative counterparts, extra critic rows/time, first KL-stop epoch/minibatch,
observed KL and configured/completed epochs. Final exact tags/schema come with
source handoff. Counter meanings are optimizer participation, not a claim that
every parameter changed numerically.

The narrow validator should check actor=joint, critic=joint+critic-only,
total=joint+critic-only, nonnegative integer counts, cumulative increments and
configured8×32 limits. Control and no-KL runs must have zero critic-only updates.
Bind every receipt to rollout step/source hash, and resolve explicitly whether
legacy `audit/optimizer_steps` means joint or total. Never infer actual calls
from an epoch bound. Preserve actor/logstd, KL, gradient/clip, value-loss and full
phase timing diagnostics; overall child/wrapper time includes all extra work.

CPU admission gates: bitwise actor mean/logstd **and actor Adam moments/step counters**
freeze during critic-only updates; gradients are `None` rather than zero tensors;
actor/critic parameters are disjoint; no-KL control parity; correct full-eight-epoch
and partial-deadline counts; fixed detached rollout targets; no extra environment
transitions. Critic-only clipping uses the existing .5 limit on critic parameters,
while ordinary joint clipping remains unchanged. Extra critic RNG must not consume
the actor's action/minibatch stream. Biased targets can make more value fitting
harmful; decreasing training MSE is not evidence of better task success.

## Minimal adapter delta / budget

Clone only the existing precision registry/command/selector/lease constants and
add a small role-counter receipt validator. Retain the same isolated images,
Volume inventories, converter/evaluator handles and download/cleanup procedure.
Use separate pair output Volumes/once-keys. No new orchestration framework or GPU
smoke is proposed. Source hashes, exact receipt schema and root's outcome rule
are the only remaining scientific admission inputs.

At documented existing rates:2×(660+30)+4×(120+30)=**1,980 GPU-container seconds**;
four CPU conversions and3,600s coordination give **$1.2655518** resource subtotal,
leaving **$1.7344482** under a proposed **$3** cap. This is not yet reserved. Root
must include the one proposed reservation in the shared $100 ledger alongside all
active/pending work before admission; this note changes no financial allowance.

Only compare policies at prespecified common measured steps. If either run misses
10.24M, show its actual steps/time and use the5.12M common point only if both reached
it; do not add post-hoc matching evaluations or extrapolate a sample-efficiency
win. Target quality remains the independently confirmed reference near96%.
A one-seed learning-curve pilot cannot establish reliable step reduction; any
expansion/confirmation requires a later root decision and separate budget.
