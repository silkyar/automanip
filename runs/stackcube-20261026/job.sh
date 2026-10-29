#!/usr/bin/env bash
set -euo pipefail
OUT=/workspace/automanip-stackcube-20261026
ROOT="$OUT/source"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/batch.exit"' EXIT
cd "$ROOT"
python3.11 - <<'PY'
import hashlib,json
from pathlib import Path
manifest=json.loads(Path('source-manifest.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
assert not Path('.env').exists()
print('Source hashes verified:',len(manifest),flush=True)
PY
timeout --signal=TERM --kill-after=30s 1800 bash deploy/runpod/bootstrap-stackcube.sh "$ROOT" "$OUT" > "$OUT/setup.log" 2>&1
export PYTHONPATH="$ROOT/src" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MS_ASSET_DIR=/workspace/mani_skill_assets MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
PY=/opt/automanip-stackcube-venv/bin/python
timeout --signal=TERM --kill-after=30s 1200 "$PY" scripts/run_stackcube_baseline.py \
    --output-root "$OUT/smoke" --smoke --seeds 1 --execute > "$OUT/smoke.log" 2>&1
timeout --signal=TERM --kill-after=30s 24000 "$PY" scripts/run_stackcube_baseline.py \
    --output-root "$OUT" --seeds 1 2 3 --execute > "$OUT/baseline.log" 2>&1
