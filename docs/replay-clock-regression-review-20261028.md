# SAC recipe-clock regression: independent review

**Recommendation:** the proposed gamma-only restoration is a reasonable single, bounded **sufficiency screen**. It can establish whether gamma .9 restores early competence inside the fast recipe. It cannot identify target-network relaxation as the cause, establish why historical SAC learned, or validate a new learning principle. No run is admitted by this review.

## What the archive establishes

At 102,400 transitions, historical SAC had 49,216 critic/actor/temperature updates, alpha .002519, and development grasp/lift/transport counts 128/115/68 of 128. Fast target-SAC had 6,088 critic and 2,029 actor/temperature updates, alpha .233609, and zero grasps. Historical SAC still achieved zero final stacking successes: early competence is the appropriate target.

The historical recipe used gamma .9, tau .01, UTD .5, actor delay 1, LR .0003, Adam beta1 .9, and three hidden critic layers. Fast SAC uses .99/.005/1⁄16/3/.001/.5/two layers. Additional differences include minibatch 1024→2048, environments 32→64, configured warmup 4000→5000 (fast actual 5056), collection 64→128, and replay capacity. Historical temperature updates resample after the actor step; fast updates reuse the actor step's pre-update log probability. These are genuine confounds, although the earlier source review found no temperature-sign, broadcasting, or timeout/export contradiction.

The initial **deterministic safe actor** digest matches (`6f768e63…`); this digest alone does not certify stochastic logstd, critic, optimizer, or random-stream identity. Historical and fast development comparisons therefore establish a recipe regression, not an isolated mechanism.

## How much weight to give the clock model

The [scalar gate](../research/discovery/replay-clock-probe-20261028.json) took .1256 CPU seconds and zero robot transitions. Under its instantaneous scalar-critic, fixed-policy assumptions, target relaxation is `1−tau*(1−gamma)` per critic update. In interaction units, historical versus fast half-lives differ about 160×; restoring gamma alone leaves 16×. The actor/temperature `LR*updates/transition` proxy remains 7.2× smaller. This proxy is not an Adam-equivalent optimization budget; doubled minibatches also make update counts distinct from sampled-row work.

The real fast target mean is 64.47 at 102,400 versus historical 3.50, while alpha is about 93× larger. Thus the scalar model's reward-only fixed point does not quantitatively explain observed soft values. Entropy bonuses, changing policy, approximation, and replay distribution can dominate. Larger values are consistent with an offset/conditioning problem, but do not prove it; Q-loss magnitudes across these scales are not comparable competence measures.

## Narrow next test and interpretation

If one robot experiment is selected, retain exactly the fast target-SAC recipe and restore **only gamma .9**, fresh training seed 1, requested 256,000 transitions/240 seconds, with cached gamma .99 development reports at 51,200/102,400/256,000. Freeze source/config differences and comparator checkpoint digests before launch; existing 102,400/256,000 weights are `851cdaac…`/`97466cba…`. Preserve full wall time, actual steps, alpha, update/row counts, and source receipts. Three checkpoint evaluations suffice; no interpolated or replacement points if censored.

At 102,400, grasp≥64/128 **and** lift≥32/128 supports discount restoration being sufficient for substantial early recovery. Both≤8/128 weakens that sufficiency hypothesis within this budget. Everything else is inconclusive. This is an exploratory, single-training-seed gate, not a significance test. Final and 51,200 results describe timing/persistence; they cannot rescue a failed primary gate through selection.

A positive result cannot separate shorter task horizon, smaller soft-value offsets, easier bootstrapping, or their downstream effects on temperature/exploration. A negative result leaves slow actor/temperature and target clocks plausible; it does not exonerate gamma or show replay learning intrinsically fails. No stacking, sample-efficiency superiority, or joint/split-normalization claim follows.

No equally cheap single-factor alternative is unambiguously more decisive for **discount sufficiency**. Restoring tau targets EMA tracking more directly but leaves discount/value scale and actor clocks untouched; restoring actor delay also changes temperature cadence and costs more updates. Neither identifies the whole historical regression. Stop after this screen rather than automatically restoring further factors.

## Reward centering and failed probes

[Naik et al., *Reward Centering*](https://arxiv.org/html/2405.09999), read through methods, control experiments and limitations, separates the state-independent `r/(1−gamma)` offset from relative values. Its continuing-task evidence makes offset conditioning a distinct plausible explanation; it does not establish neural SAC/StackCube causality. Simple off-policy reward averages can estimate the wrong policy, and the paper explicitly warns that episodic centering can alter the problem. No centering implementation is recommended here.

Both continuation-noise setups stopped before any attempted simulator transition. The v2 physical-state residual passed, but derived observations exceeded the frozen tolerance. These are setup-validity failures, with no variance or learning conclusion; no third setup is proposed.

Sources: archived historical `summary.json`, `development-diagnostic-summary.json`, `sac/native/trainer.py`, `run/run.json` and `training.jsonl`; fast `target-sac/target-sac/run/{report.json,learning.jsonl}` and archived learner/robot sources. No protected episode outputs, new probes, training, or cloud calls were used.
