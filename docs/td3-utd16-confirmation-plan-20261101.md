# Once-only confirmation of the td3-utd16 StackCube recipe (prepared, not run)

Registered before any confirmation query. **Status: approved one-time by the repository owner on Nov 1 and run once; target met (mean 95.39%, minimum seed 94.34%).** See the [results](sample-efficiency-results-20261031.md#protected-confirmation-of-td3-utd16-once-only-root-approved--target-met). The research agent wrote this package; under the [earlier confirmation protocol](stackcube-precision-confirmation-plan-20261028.md) root must review the admission, contract and driver, and only root sets `root_reviewed` and `launch_enabled`.

**Policies (fixed).** The final 7.68M export of every fresh-seed r4 run (seeds 21–25), selected by rule before any confirmation query, so no seed is cherry-picked: [candidates](td3-utd16-confirmation-candidates-20261031.json). Local copies: `runs/sample-efficiency-r4-20261031/td3-utd16-seed-*/step-7680000`; digests were verified against the volume on Nov 1. Public-dev successes 124, 127, 122, 121, 124 /128.

**Scope.** 512 frozen confirmation episodes per policy (2,560 total), with the unchanged frozen evaluator function (`deploy/modal/evaluate.py` sha256 `83179bb0…`, the same bytecode extraction as the 50M confirmation), the same benchmark protocol `fe2cc598…`, task, validator, pins, horizon and success definition. At most 3 L40S scorers, 220 s evaluator limit, 240 s container limit, no retries, a 1,800 s CPU coordinator, and GPU attempt keys that prevent repeats. Budget cap $2 (resource bound ≈$0.86 = 5 × 240 s L40S plus the coordinator).

**Target.** Mean success at the end of the episode ≥95%, and every seed ≥90%. These are lower than the 50M recipe's 99%/98% target because this recipe's claim is sample efficiency with performance in the 90s, not maximal precision. Failure is recorded as such and is not repaired by another query. All counts are reported either way.

**Files.** `deploy/modal/td3-confirmation-plan.json` (disabled), `deploy/modal/td3_confirmation_contract.py`, `deploy/modal/td3_confirmation.py` (a mechanical copy of `precision_confirmation.py`; only the plan, contract, names and policy count differ), `tests/test_modal_td3_confirmation.py` (5 tests including "committed plan is denied"; passing).

**To run (root only), after review:** set `launch_enabled`, `root_reviewed`, `root_review_at` and `root_review` in the plan JSON, then `MODAL_CONFIG_PATH=$PWD/.runtime/modal-config.toml .venv-modal/bin/modal run deploy/modal/td3_confirmation.py::confirm_td3`.
