# Shared paths for simulation/*.sh (source this file).
# shellcheck shell=bash
_SIMULATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LBM_ROOT="$(cd "$_SIMULATION_DIR/.." && pwd)"
THIRD_PARTY="${LBM_ROOT}/third_party"
DATASETS="${LBM_DATASETS:-${LBM_ROOT}/datasets}"
LEROBOT_DIR="${LBM_ROOT}/simulation/lerobot"
LIBERO_ROOT="${LIBERO_ROOT:-${THIRD_PARTY}/libero}"
RMBENCH_ROOT="${RMBENCH_ROOT:-${THIRD_PARTY}/rmbench}"
