# Third-party attribution

`train.py` adapts the state-based PPO baseline from ManiSkill v3.0.0b21:
https://github.com/haosulab/ManiSkill/blob/v3.0.0b21/examples/baselines/ppo/ppo.py

ManiSkill contributors distribute that work under Apache License 2.0; a copy is
in `third_party/ManiSkill-LICENSE`. AutoManip changes include bounded config and
budgets, explicit bootstrap helpers, safe actor exports, local CPU compatibility,
run provenance, and separate evaluator execution. It is not an unchanged copy
or a claim of matching trained benchmark performance.

`third_party/maniskill/ppo.py` and `examples.sh` are unmodified copies from
ManiSkill commit `6226567ce326625fa6eacf9aeb3f4008e4339f7e` (v3.0.0b21),
under the same Apache 2.0 license. StackCube's first reference uses the official
25M-step example with operational output/video flags, not the separately listed
50M-step `ppo_fast.py` published benchmark recipe.
