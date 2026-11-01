# Local reuse-and-adaptation milestone

Date: 2026-10-29. Local implementation and CPU validation only. No simulator was constructed, no new robot experience was collected, and no cloud/GPU training or benchmark scoring was performed for this milestone.

Follow-up: the user authorized the [2026-10-30 GPU preflight](cumulative-preflight-results-20261030.md). It found that the source already passes the proposed target, repaired a new-wrapper batching fault, and stopped before training. The dated local-only results below describe the earlier milestone.

The path now supports taking an existing safe StackCube actor, performing a bounded ordinary PPO adaptation attempt in a supplied compatible environment, preserving its new experience and parent lineage, and assessing the same adapted actor on both the target and original distributions. **This demonstrates local plumbing, not a measured transfer benefit or cumulative-learning result.**

## Review reconciliation

The [review reconciliation](cumulative-review-reconciliation-20261029.md) checks both downloaded documents against the current working tree. The main architectural findings stand. Important corrections are that usable actor/data assets exist locally despite their exclusion from the GitHub snapshot; actor loading is already supported but is not full learner resume; and two diagnostics described as pending in the review have since completed, with negative or conditional evidence. The selected actor's historical confirmation is 497/512 (97.07%); 95.90% is the three-seed recipe mean.

The first-milestone brief governs this work. The older roadmap is context, not permission to build all its components or resume paid experiments.

## Code and boundaries

| File | Responsibility |
|---|---|
| [cumulative.py](../src/automanip/cumulative.py) | Explicit robot/controller/units/frames/timing/observation/action contracts, separate goal identity, safe actor asset registration, parent-cost accounting and paired same-controller retention gates. |
| [cumulative_ppo.py](../src/automanip/cumulative_ppo.py) | Actor-only warm-start or scratch initialization; existing `train.py` GAE, successor handling and PPO update; bounded collection/updates/feedback; safe export and partial/error accounting. |
| [cumulative_episodes.py](../src/automanip/cumulative_episodes.py) | Episode recorder and bounded data-only loader; per-lane boundaries, actual successors, clipped executed actions, terminal versus timeout flags, behavior revisions and partitions. Imports the known existing collector format without selecting only successes. |
| [cumulative_env.py](../src/automanip/cumulative_env.py) | Closed reset-only placement variation, lazy future simulator factory and assessment through the existing evaluator kernel. Synthetic tests cover arithmetic/API order only. |
| [cumulative_plan.py](../src/automanip/cumulative_plan.py) | Exact source registration and non-executing experiment compiler, including source-code hashes, controls, seeds, gates and budgets. |
| [prepare_cumulative_transfer.py](../scripts/prepare_cumulative_transfer.py) | Local preparation CLI with no launch option. Existing output directories are rejected. |
| [source actor manifest](../cumulative/assets/stackcube-50m-seed1-v1.json) | Immutable reference to the existing safe actor and its acquisition/applicability evidence. |
| [experience reference](../cumulative/assets/source-training-episodes-v1.json) | Location/provenance of the converted existing training episodes; unused by this pilot's PPO strategy. |

The deterministic actor is restored. The critic, Adam state, exploration scale (Gaussian log-standard-deviation −0.5) and attempt RNG are fresh; there is no normalizer or replay restoration. No source critic/reward is reused for a new goal. This is ordinary fine-tuning, not exact learner/simulator continuation. Deployment remains the existing restricted JSON MLP; no new runtime architecture, candidate-code execution or evaluator pickle loading was added.

Attempts preserve complete failures and incomplete episodes, reserve work before external calls, distinguish attempted/completed/uncertain transitions and updates, and record feedback costs even when feedback is invalid. A flushed journal can recover work reservations after process interruption; it cannot recover the learner/simulator or guarantee persistence across power loss. Storage rotates into bounded episode shards. Development and final data are rejected as training stores.

`adapt()` owns and closes the supplied environment, including rejected inputs and setup errors. Its cooperative inner clock excludes caller imports, asset parsing and environment construction. A future outer process deadline must cover those and enforce the registered total window. The current implementation does not establish an end-to-end ten-minute result or adversarial isolation.

## Checks actually run

Final selected CPU command:

```sh
PYTHONWARNINGS=ignore PYTHONPATH=src:. .venv311/bin/python -m unittest \
  tests.test_cumulative tests.test_cumulative_episodes tests.test_ppo \
  tests.test_core tests.test_rollout tests.test_stackcube -q
```

**77 tests passed in 1.244 seconds.** These use temporary artifacts, synthetic tensors and fake environments/adapters. Coverage includes incompatible semantic interfaces, safe artifact validation, actual PPO updates on a fake environment, fresh-state initialization, source immutability, incomplete/error accounting, update/time/interaction ceilings, interrupted-update reservations, feedback cost/partition checks, no extra final reset, same-controller retention, and the existing evaluation kernel on a synthetic adapter. This is a selected suite, not a claim that every repository test was run. Log: `.runtime/cumulative-test-final.log`.

Two additional checks used real local assets without training or simulator collection:

- Loaded the existing source actor into `AdaptationAgent` and compared four already-recorded public-training observations with trusted NumPy inference. Maximum absolute action difference was **2.98e−7**; fresh critic/exploration state was recorded. No learner update was performed in this check. Receipt: `.runtime/cumulative-actual-source-parity.json`.
- Imported and reloaded **512 old training episodes / 25,600 transitions**, preserving **497 successes and 15 failures**. This created no new experience or teacher labels. Import plus reload took approximately 1.435 seconds. Store: `artifacts/cumulative/source-episodes-v1/`; sibling import receipt binds all store files. Receipt: `.runtime/cumulative-actual-import-validation.json`.

The legacy importer accepts only the pinned collector contract. Its termination/truncation reconstruction is explicitly derived from that collector's full-50-step/no-autoreset behavior, not claimed to have been stored originally. The legacy reader requires at least four successful episodes; arbitrary all-failed legacy datasets are unsupported. New episode stores preserve failures directly and enforce bounded, non-pickle loading.

SHA256 verification against the pre-change baseline found **all 1,684 protected files (1,038,085,022 bytes) unchanged**, with no new files under the historical benchmark/run/dependency/vendor prefixes. Frozen policy loading, task definitions, historical evaluator files and historical results were not edited. Receipt: `.runtime/cumulative-milestone-integrity.json`. Unrelated pre-existing working-tree changes remain in place.

## Prepared experiment and remaining work

The [exact plan JSON](../cumulative/experiments/stackcube-north-pilot-v1.json) and [scientific design](cumulative-transfer-plan-20261029.md) specify three controls: unchanged source, scratch PPO and ordinary actor warm-start. The target shifts the stacking destination beyond the original sampler's y support while preserving the original controller, observations, physical success and reward. Assess the **same adapted actor** in both domains; routing to the archived source earns no retention credit.

The pilot uses learner seed 41, a ceiling of **2.048M new training transitions / 600 seconds per trainable arm**, and development checkpoints at 0, 128k, 512k, 1.024M and 2.048M transitions. The joint target is ≥90% target success and ≥90% original-task success with at most five percentage points of original-task loss against the paired unchanged reference. Independent final seeds are separate from checkpoint selection. Source acquisition costs remain visible; the existing demonstrations are not used for PPO learning. Replication slots 42/43 are specified but unlaunched.

Prepared output is `artifacts/cumulative/pilot-preparation-v2/`, state `prepared_not_executed`, execution authorization false. Version 1 remains preserved but is stale after implementation fixes. To create another local preparation in a **new** directory:

```sh
PYTHONPATH=src:. .venv311/bin/python scripts/prepare_cumulative_transfer.py \
  --output artifacts/cumulative/another-preparation
```

There is no missing local source actor or source dataset. A fresh checkout without ignored bulk files will need those exact assets; preparation reports their absence instead of fabricating them. Target-task trajectories, adapted robot policies, learning curves and fresh retention/final scores do not yet exist because their collection is not authorized.

Before any robot execution, validate target reachability/contact/observation behavior, real GPU pose synchronization and actual throughput, and implement/validate the outer hard deadline plus independent final-assessment admission/isolation. Synthetic tests do not settle those questions. No corrective controller, action-consequence model, generic selector, A→B→C roadmap or tenfold sample-efficiency claim is implemented or established here.
