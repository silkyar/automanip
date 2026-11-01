# SAC first-smoke failure and replay dtype fix — 2026-10-27

The trusted owner reports that the first robot smoke failed on its first vector
step, before any learner update. Actual simulator experience was **8 transitions,
zero gradient updates**. The failed attempt is preserved; it is not a successful
800-transition smoke or a policy-performance result. This agent did not inspect
private outputs, launch that attempt, or rerun it.

**Observation.** `transition_successor` correctly returns a boolean true-terminal
mask. Replay stores the Bellman stop coefficient in float32. Advanced indexed
assignment requires matching dtypes, so assigning Bool directly into Float storage
raised `Index put requires the source and destination dtypes match`.

**Mechanism/fix.** `Replay.add` now detaches each incoming tensor and explicitly
converts it to the destination tensor's device and dtype before checking finite
values and storing it. A true terminal becomes float 1; truncation/continuation
become 0. The final pre-reset observation and bootstrapping rule are unchanged.
Checking finiteness after conversion also rejects overflow caused by narrowing.
No algorithm, reward, evaluator, export format or comparison protocol changed.

**Prior art / novelty.** Engineering correction only. No learning-mechanism or
novelty claim.

**Prediction.** Inserting the actual boolean terminal mask into float replay no
longer raises, and a truncated transition still bootstraps from its final state.

**Falsification/test.** New regression constructs the output of
`transition_successor` for a truncation, a true terminal and a continuation, then
passes that exact boolean mask into replay. It verifies stored stop values
`[0,1,0]` and next observations `[7,8,33]`, rather than reset observations.
The earlier tests had covered successor handling and replay independently with
float stop inputs; they missed this integration boundary.

**Outcome.** All **17 discovery tests pass** on the existing CPU torch environment
(0.90 seconds measured test runtime). No additional simulator experience, GPU time,
teacher data or paid APIs were used for this correction. Actual cost of the
owner-run failed smoke remains in the owner's accounting; it is not assumed zero.

Old source SHA256:
`91e4aefd87371490e65f857d62ff69777d32c89f13491d03afd2679cd359db91`.

Fixed source SHA256:
`1191b6bafb1c5a34cb2f661a23da732cf2021ec680f3971477d4c02fa8f614ca`.

**Next decision.** The owner may separately admit one smoke for this new code
version after review. The old attempt and its budget remain separate. This agent
makes no launch and does not amend the owner's wrapper. The successful-smoke
expectation remains 800 transitions, 100 vector calls, 16 completed episodes,
24 instrumented reset episodes, three reset calls and 93 updates of each optimizer.

```sh
.venv311/bin/python -m unittest discover -s tests -p 'test_discovery_*.py' -v
```
