#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?source directory required}
OUT=${2:?output directory required}
cd "$ROOT"
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/setup.exit"' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libegl1 libgl1 libxext6 libvulkan1 vulkan-tools
python3.11 -m venv /opt/automanip-stackcube-venv
PY=/opt/automanip-stackcube-venv/bin/python
"$PY" -m pip install --no-cache-dir --only-binary=:all: --require-hashes \
    -r requirements/linux-cu124.txt -r requirements/stackcube-extra.txt
"$PY" -m pip check
"$PY" -m pip freeze --all > "$OUT/installed.txt"
dpkg-query -W > "$OUT/os-packages.txt"
nvidia-smi -q > "$OUT/nvidia-smi.txt"
export PYTHONPATH="$ROOT/src" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MS_ASSET_DIR=/workspace/mani_skill_assets MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
"$PY" -m unittest discover -s tests -v
"$PY" scripts/check_stackcube.py
echo STACKCUBE_SETUP_PASSED
