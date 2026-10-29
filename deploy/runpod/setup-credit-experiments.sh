#!/usr/bin/env bash
set -euo pipefail
OUT=${1:?persistent campaign directory required}
ROOT="$OUT/source"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/setup.exit"' EXIT
cd "$ROOT"
mkdir -p "$OUT/bootstrap"
python3.11 - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('source-manifest.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in m.items())
print('Source manifest verified:',len(m),flush=True)
PY
timeout --signal=TERM --kill-after=30s 1800 bash deploy/runpod/bootstrap-stackcube.sh "$ROOT" "$OUT/bootstrap"
LOCAL=/opt/automanip-credit-20261026
mkdir -p "$LOCAL/source" "$LOCAL/assets"
cp -a "$ROOT/." "$LOCAL/source/"
if test -d /workspace/mani_skill_assets; then cp -aL /workspace/mani_skill_assets/. "$LOCAL/assets/"; fi
export PYTHONPATH="$LOCAL/source/src" PYTHONDONTWRITEBYTECODE=1
cd "$LOCAL/source"
/opt/automanip-stackcube-venv/bin/python - "$OUT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.run_stackcube_experiments import prepare
prepare(Path(sys.argv[1])/'permission-preflight')
print('EXPERIMENT_ENVIRONMENT_READY',flush=True)
PY
