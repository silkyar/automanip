# AutoManip

A small foundation for autonomous experiments on robot policies. ManiSkill and
PyTorch are the intended simulator/training stack. No Tinker service is needed.

## Run the CPU checks

From this directory, with Python 3.11 or newer:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The core and tests require no third-party packages. No simulator or GPU tests
have been run. Optional simulator versions in `pyproject.toml` are a provisional,
pinned starting recipe, not a resolved or tested dependency lock.

## Code

- `src/automanip/task_spec.py`: frozen task fields, fingerprint, backend constraints.
- `src/automanip/policy_io.py`: bounded data-only policy validation.
- `src/automanip/metrics.py`: stock Panda PickCube rule, separate placement settling rule,
  and Wilson confidence intervals.
- `src/automanip/evaluator.py`: policy/action/result contracts, without simulator rollouts.
- `tests/test_core.py`: isolated temporary artifacts and CPU unit tests.
- `docs/upstream.md`: verified APIs, reference differences, and open verification work.

## Artifact interface

An artifact directory has fixed names `manifest.json` and `weights.json`.
The manifest has exactly these fields:

```json
{
  "format": "automanip-json-f32-v1",
  "task_fingerprint": "<trusted task fingerprint>",
  "observation_dim": 42,
  "action_dim": 7,
  "hidden_sizes": [256, 256, 256],
  "activation": "tanh",
  "weights_sha256": "<SHA-256 of exact weights.json bytes>"
}
```

**42 is illustrative**, not a verified observation dimension. The evaluator must
measure and freeze the pinned environment's observation interface before accepting
real candidates. It supplies `PolicySchema`; candidates cannot choose its dimensions.

Weights map `layers.N.weight` and `layers.N.bias` to `{"shape": [...], "values": [...]}`.
Weights are row-major `[output, input]`; biases are `[output]`. Hidden layers use
tanh, the final layer is linear. Trusted inference must clip means to normalized
action bounds before `validate_actions`; nonfinite outputs must fail the episode.
Only deterministic actor parameters are accepted, not critic/optimizer state.

The loader checks exact fields and tensor names, dimensions, finite float32 values,
byte/parameter limits, and digest. It never executes candidate code. JSON was chosen
for dependency-free CPU validation; safetensors is the intended efficient replacement
before full training. Digests identify bytes, not trustworthy provenance.

## Boundaries and next milestones

This is a local prototype, not an adversarial security boundary. A research agent
with unrestricted shell access could modify these files. Unattended runs require
separate trainer/evaluator identities, mounts, and processes or VMs, with private
seeds inaccessible to the trainer.

Next: verify the full pinned simulator source and resolve dependencies on Linux;
measure observation/action spaces; add trusted model inference and vectorized
rollouts with correct final observations and termination/truncation handling.
Then port PPO training from random initialization. The custom placement checker
is a tested rule only: an evaluator-owned table-placement task and contact-state
adapter still need implementation. No autonomous loop, training claims, or private
holdout results are included in this milestone.
