#!/usr/bin/env bash
# Dataset-local frozen visual features. Forward all training/prebuild flags.
# Example: bash scripts/prebuild_visual.sh --dataset agibot --max-episodes 4 --no-mmap
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${LBM_PYTHON:-python}" "${SCRIPT_DIR}/prebuild_features.py" "$@"
