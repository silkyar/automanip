#!/usr/bin/env bash
set -euo pipefail
OUT=/workspace/automanip-research-20261026-credit-v6
ROOT="$OUT/source"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/batch.exit"' EXIT
cd "$ROOT"
python3.11 - <<'PY'
import hashlib,json
from pathlib import Path
manifest=json.loads(Path('source-manifest.json').read_text())
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in manifest.items())
assert not Path('.env').exists()
print('Research source manifest verified:',len(manifest),flush=True)
PY
mkdir -p "$OUT/bootstrap"
if test -d /workspace/automanip-stackcube-20261026; then
    chmod 0700 /workspace/automanip-stackcube-20261026
fi
if timeout --signal=TERM --kill-after=30s 1800 bash deploy/runpod/bootstrap-stackcube.sh "$ROOT" "$OUT/bootstrap" > "$OUT/setup.log" 2>&1; then
    printf '0\n' > "$OUT/setup.exit"
else
    rc=$?; printf '%s\n' "$rc" > "$OUT/setup.exit"; exit "$rc"
fi
chmod 0700 "$OUT/baseline-development"
export PYTHONPATH="$ROOT/src" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MS_ASSET_DIR=/workspace/mani_skill_assets MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export PYTHONDONTWRITEBYTECODE=1
LOCAL=/opt/automanip-credit-20261026
mkdir -p "$LOCAL"
mkdir -p "$LOCAL/source"
cp -a "$ROOT/." "$LOCAL/source/"
if test -d /workspace/mani_skill_assets; then
    cp -aL /workspace/mani_skill_assets "$LOCAL/assets"
else
    mkdir -p "$LOCAL/assets"
fi
chmod 0700 "$OUT" "$ROOT"
ROOT="$LOCAL/source"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
PY=/opt/automanip-stackcube-venv/bin/python
timeout --signal=TERM --kill-after=30s 1200 "$PY" scripts/run_stackcube_experiments.py \
    --output-root "$OUT/smoke" --smoke --execute > "$OUT/smoke.log" 2>&1
timeout --signal=TERM --kill-after=30s 24000 "$PY" scripts/run_stackcube_experiments.py \
    --output-root "$OUT" --execute > "$OUT/experiments.log" 2>&1
