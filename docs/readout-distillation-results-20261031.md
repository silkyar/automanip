# Most search improvement did not survive a final-layer student fit

Four fixed CPU readout fits produced safe MLP artifacts. The scorer accepted
only those numeric artifacts, contained no fitting corpus/proposal module, and
used the unchanged rollout kernel and physical success rules. Every policy
acted alone from reset for 50 steps on fresh 962-series public episodes.

| Deterministic policy | Parent 2 /512 | Parent 3 /512 | Total /1024 |
| --- | ---: | ---: | ---: |
| Original weak parent | 1 | 0 | 1 (0.10%) |
| Search-data student | 9 | 25 | 34 (3.32%) |
| Unbranched-data student | 5 | 9 | 14 (1.37%) |

Search minus original gained 3.22 percentage points,97.5%paired block-bootstrap
interval[2.25,4.30]. Search minus unbranched gained 1.95 points,[0.59,3.42]. These
are positive conditional differences but fail the predeclared material screens
of 10 and 5 points. No threshold was relaxed and no model was promoted. The best
historical 95.9%confirmation result remains unchanged. This is assisted adaptation
from two fixed acquired parents, not a population replication, held-task result
or from-scratch sample-efficiency improvement.

Search-student ever-grasp was 508/512 and 502/512, passing the 95%grasp screen. That
does not establish retention of all prior behavior: placement declined from
506/505 episodes to 474/471. Release remained the main problem:26/50 search-student
episodes ever released over the target, versus 3/2 for originals. Most failures
still ended in the `placed_still_grasped` diagnostic. Stability-for-ten-steps
remains a diagnostic, not an extra success requirement.

The unbranched control received 115,200 newphysics transitions, equal to the full
coordinate study including its diagnostic arm/repeats. It sampled one relative
continuation per independent episode, without comparing futures:181/1,152 and
130/1,152 successful training continuations. It had more distinct source states
than search. The earlier 229,120 branching diagnostics/search interactions and
the original parent acquisition remain additional disclosed costs. This design
does not support a 10 x learning claim.

Each fit used 384 teacher and 1,024 early-rehearsal rows from a 75/25 episode split.
Both methods used the same closed-form objective, rehearsal weight 4 and ridge
0.001, without tuning on public outcomes. Four 257 dimensional solves took
0.648 CPU seconds, including safe export;7,424 hidden-forward rows include fitting,
internal validation and export checks. No PPO/GPU optimizer updates. Offline
teacher-action losses decreased, but were not treated as policy scores.

A registered CPU-only postmortem exposes the precision/retention bottleneck.
On all held-out teacher episodes, registered students' finger-target RMSE was
13.23/10.29 mm, while median teacher increments were only 0.276/0.495 mm. The
students requested positive increments in 7.31%/1.67%of those states, versus
76.92%/66.67%for teachers. Average 8-coordinate loss obscured this difference.
Removing rehearsal in a diagnostic-only solve improved late prediction but
increased early action RMS change to 0.623/0.586. No such diagnostic model was
exported or scored. An additional linear probe of the measured-finger hold
coordinate still had 3.88/3.64 mm error under fixed ridge/features. This suggests
investigating representation precision and interference, not simply increasing
the number of sampled futures. It does not prove a capacity lower bound or a
physical causal explanation for every failed rollout.

Accounting:115,200 control-collection and 153,600 fresh public assessment transitions;
four original CPU readout solves, plus four separately registered postmortem
solves(0.141 seconds, zero physics/exports). Cumulative branching investigation
**613,120 of 1 Mtransitions**. Control wrappers 44.72/44.51 seconds; scorer wrappers
70.16/77.67 seconds. Control and scorer archives contain 46/26,050,631 bytes and
22/2,007,408 bytes respectively, plus inputs/sources. Both apps read back stopped
with zero tasks. No confirmation/final query or score-driven refit.

[Protocol](../cumulative/experiments/readout-distillation-v1.json);
[assessment](../cumulative/results/readout-score-20261031.json);
[CPU postmortem](../cumulative/results/readout-residual-diagnosis-20261031.json).
