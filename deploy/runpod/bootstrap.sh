#!/usr/bin/env bash
# Install the reviewed Linux lock on an official Python 3.11 Runpod image.
set -euo pipefail
cd /workspace/automanip/source
mkdir -p /workspace/automanip/evidence
trap 'rc=$?; printf "%s\n" "$rc" > /workspace/automanip/evidence/setup.exit' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libegl1 libgl1 libxext6 libvulkan1 vulkan-tools
python3.11 -m venv /opt/automanip-venv
/opt/automanip-venv/bin/python -m pip install --no-cache-dir --only-binary=:all: \
    --require-hashes -r requirements/linux-cu124.txt
/opt/automanip-venv/bin/python -m pip check
/opt/automanip-venv/bin/python -m pip freeze --all > /workspace/automanip/evidence/installed.txt
dpkg-query -W > /workspace/automanip/evidence/os-packages.txt
nvidia-smi -q > /workspace/automanip/evidence/nvidia-smi.txt
export PYTHONPATH=/workspace/automanip/source/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
/opt/automanip-venv/bin/python -m unittest discover -s tests -v
/opt/automanip-venv/bin/python -c 'import torch; assert torch.__version__.split("+")[0] == "2.6.0"; assert torch.cuda.is_available(); x=torch.ones(16,device="cuda"); assert x.sum().item()==16; print("CUDA arithmetic passed:", torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)'
echo SETUP_PASSED
