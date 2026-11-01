#!/usr/bin/env bash
set -euo pipefail
OUT=${1:?persistent campaign directory required}
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/batch.exit"' EXIT
bash "$OUT/source/deploy/runpod/setup-credit-experiments.sh" "$OUT"
ROOT=/opt/automanip-credit-20261026/source
cd "$ROOT"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MS_ASSET_DIR=/opt/automanip-credit-20261026/assets MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
timeout --signal=TERM --kill-after=30s 26000 /opt/automanip-stackcube-venv/bin/python \
  scripts/run_settling_campaign.py --output-root "$OUT"
