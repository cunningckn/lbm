#!/bin/bash
# Fine-tune LBM on the converted RMBench LeRobot dump.
#
# Usage (from lbm root):
#   bash simulation/rmbench/finetune.sh
#   NPROC=4 bash simulation/rmbench/finetune.sh --fsdp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

export DATASET="${DATASET:-rmbench}"
export ROBOT_TYPE="${ROBOT_TYPE:-rmbench}"
cd "${LBM_ROOT}"
exec "${LBM_ROOT}/scripts/train.sh" "$@"
