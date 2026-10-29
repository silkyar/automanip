# Adapted SAC control: no successful stacks in this screen

The seed-1 adapted SAC policy scored **0/128 final-step development successes**
at all four preselected checkpoints. Training stopped at its original deadline:
262,432 actual transitions and 1,292.19 trainer seconds (21.54 minutes), below
the requested 500,000-transition ceiling. No confirmation query or additional
training seed was launched. This recipe is shelved while the quality campaign
focuses on the strong PPO reference; it does not refute SAC in general.

| Training transitions | Checkpoint ready after | Successful stacks | Ever placed |
|---:|---:|---:|---:|
| 51,200 | 4.34 min | 0/128 | 0/128 |
| 102,400 | 8.59 min | 0/128 | 2/128 |
| 256,000 | 20.99 min | 0/128 | 51/128 |
| 262,432 | 21.51 min | 0/128 | 55/128 |

The final policy grasped in 128/128, lifted in 115/128, transported in 112/128,
and placed in 55/128 development episodes, but released in none. It learned
meaningful parts of the task without completing the stack. These stage-ever
diagnostics do not identify the cause or establish an ordered sequence.

The run performed 129,216 joint-critic, actor and entropy-temperature optimizer
rounds each; joint-critic rounds train two Q networks. It drew 132,317,184 replay
minibatch rows, completed 5,248 training episodes and recorded 5,280 reset episodes
including initialization. No training-side evaluation, teacher, demonstrations,
pretraining or planning rollouts were used. Four separate frozen development
evaluations added 512 episodes / 25,600 transitions. Training-side simulation and
reset work are timed separately, with reset time already included in step time.

The source and explicit adaptations are described in the
[screen plan](stackcube-sac-screen-20261027.md) and
[implementation journal](research-discovery-sac-baseline-20261027.md).
The actor differs from upstream SAC to fit the existing safe inference contract;
this is not an unmodified official StackCube baseline. The failed dtype smoke
and its eight transitions remain preserved separately from the corrected smoke
and this performance run.

All 34 archived files (21,302,106 bytes) passed hash/length verification, including
the seven native actor exports, source, traces and four complete frozen reports.
The app stopped automatically with zero live containers. Reported screen cost was
**$0.92470**, with possible meter lag; smoke costs were separate. Full records are
under `runs/stackcube-sac-screen-seed1-20261027/`, including the development
diagnostic summary. Summary SHA256:
`2449446dcfba1e7fe8d0d1733933dfa91c148bab3900a9e27c301284e60bd1ce`.
