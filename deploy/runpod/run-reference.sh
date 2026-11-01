#!/usr/bin/env bash
# Explicit batch job. Process deadlines do not themselves stop the billable Pod.
set -euo pipefail
cd /workspace/automanip/source
export PYTHONPATH=/workspace/automanip/source/src
export MS_ASSET_DIR=/workspace/mani_skill_assets MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PY=/opt/automanip-venv/bin/python
OUT=/workspace/automanip
mkdir -p "$OUT/evidence" "$OUT/runs"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/evidence/reference.exit"' EXIT
date -u
echo STAGE_GPU_ENVIRONMENT_CHECK
timeout --signal=TERM --kill-after=30s 300 "$PY" train.py \
    --config configs/gpu-smoke.json --backend physx_cuda --check-env
find /root/.sapien/physx -type f -exec sha256sum {} + > "$OUT/evidence/physx-sha256.txt"
echo STAGE_SMOKE_TRAINING
timeout --signal=TERM --kill-after=30s 300 "$PY" train.py \
    --config configs/gpu-smoke.json --backend physx_cuda --seed 1 \
    --output "$OUT/runs/smoke-seed-1"
"$PY" -c 'import json; from pathlib import Path; p=Path("/workspace/automanip/runs/smoke-seed-1"); r=json.loads((p/"run.json").read_text()); assert r["trained"] and r["learner_updates"] > 0 and not r["errors"]; assert (p/"policy/manifest.json").is_file()'
echo STAGE_SMOKE_EVALUATION
timeout --signal=TERM --kill-after=30s 300 "$PY" scripts/evaluate.py \
    "$OUT/runs/smoke-seed-1/policy" --episodes 8 --num-envs 8 --backend physx_cuda \
    > "$OUT/evidence/smoke-evaluation.log"
"$PY" -c 'import json; from pathlib import Path; lines=Path("/workspace/automanip/evidence/smoke-evaluation.log").read_text().splitlines(); r=json.loads(lines[-1]); assert len(r["episodes"])==8; assert all(not e["outcome"]["failed"] and e["steps"]==50 for e in r["episodes"])'
echo SMOKE_PASSED
for seed in 1 2 3; do
    echo "STAGE_REFERENCE_TRAINING_SEED_$seed"
    timeout --signal=TERM --kill-after=30s 900 "$PY" train.py \
        --config configs/baseline.json --backend physx_cuda --seed "$seed" \
        --output "$OUT/runs/reference/seed-$seed"
    echo "STAGE_REFERENCE_EVALUATION_SEED_$seed"
    timeout --signal=TERM --kill-after=30s 900 "$PY" scripts/evaluate.py \
        "$OUT/runs/reference/seed-$seed/policy" --episodes 128 --num-envs 8 \
        --backend physx_cuda > "$OUT/evidence/evaluation-seed-$seed.log"
done
date -u
echo REFERENCE_BATCH_FINISHED
