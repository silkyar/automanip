#!/usr/bin/env bash
set -euo pipefail
OUT=/workspace/automanip-stackcube-20261026
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/diagnostics.exit"' EXIT
cd "$OUT/source"
while test ! -f "$OUT/batch.exit"; do sleep 10; done
test "$(cat "$OUT/batch.exit")" = 0
export PYTHONPATH="$OUT/source/src" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MS_ASSET_DIR=/workspace/mani_skill_assets MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
/opt/automanip-stackcube-venv/bin/python - <<'CHECK'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('/workspace/automanip-stackcube-20261026/diagnostics-tools/trace_stackcube.py').read_bytes()).hexdigest() == '47b3ab8893aaaa406faf4b243342e17a6c3460eb7802a84a25c7f61aa7a708ae'
CHECK
timeout --signal=TERM --kill-after=30s 1800 /opt/automanip-stackcube-venv/bin/python "$OUT/diagnostics-tools/trace_stackcube.py" "$OUT" --output "$OUT/diagnostics"
