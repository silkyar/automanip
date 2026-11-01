#!/bin/sh
# Launch project Python with only this command's local MoltenVK settings.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export VK_ICD_FILENAMES="$project_dir/.runtime/MoltenVK/MoltenVK/dynamic/dylib/macOS/MoltenVK_icd.json"
export MVK_CONFIG_LOG_LEVEL=0
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$project_dir/.venv311/bin/python" "$@"
