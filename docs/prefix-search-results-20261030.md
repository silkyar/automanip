# Fresh-prefix search was reproducible but nearly all labels were negative

All prefix and repeated-continuation fidelity checks passed. Avoiding state
writes by replaying the original forward history in a fresh process worked on
the two tested acquired policies and 128 training source states. This is finite
evidence for that procedure, not a general simulator determinism guarantee.

The 16-candidate search covered **0/64** parent-2 states and **1/64** parent-3
states with a final-success continuation. The pre-fixed quiet and random proposal
controls both scored zero. The registered coverage screen failed, so no student
dataset was exported from this original search. No shaped reward/value model was
used and no policy was promoted.

Post-outcome inspection found the quiet-action prior was not physically neutral
for the gripper. Its normalized zero maps to 0.015 metres, not the current finger
position. Median current finger positions were about 0.018 metres. Most quiet
gripper targets narrowed the opening; the arm coordinates were joint deltas.
That is a sampler/controller-interface issue, not a general learning discovery.
The following controlled coordinate test holds all arm commands/noise fixed.

Work:115,200 newtransitions including every repeated prefix and four full repeat
checks; zero optimizer/confirmation/final queries. Two L 40 S wrappers 172.59 and
151.12 seconds.148 remote files/49,329,539 bytes were archived with exact inputs and
sources; app read back stopped with zero tasks. Four proposal tests and three
analysis tests passed, including negative prefix/coverage controls.

[Protocol](../cumulative/experiments/prefix-search-v1.json);
[result](../cumulative/results/prefix-search-20261030.json);
[coordinate followup](relative-gripper-plan-20261031.md).
