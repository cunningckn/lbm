#!/bin/bash
# Convert RMBench raw demos → one LeRobot dataset (video storage).
#
# Usage:
#   bash simulation/rmbench/convert_rmbench_data_to_lerobot.sh
#   bash simulation/rmbench/convert_rmbench_data_to_lerobot.sh 50
#   TASK_SET=m1 bash simulation/rmbench/convert_rmbench_data_to_lerobot.sh 50
#   OVERWRITE=1 bash simulation/rmbench/convert_rmbench_data_to_lerobot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

NUM="${1:-50}"
SETTING="${SETTING:-demo_clean}"
TASK_SET="${TASK_SET:-all}"
case "${TASK_SET}" in
  all) OUT_NAME="rmbench" ;;
  m1) OUT_NAME="rmbench_m1" ;;
  mn) OUT_NAME="rmbench_mn" ;;
  *) echo "ERROR: TASK_SET must be all|m1|mn (got ${TASK_SET})" >&2; exit 1 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-${DATASETS}/${OUT_NAME}}"

if [[ ! -x "${LEROBOT_DIR}/.venv/bin/python" ]]; then
  echo "ERROR: missing convert env at ${LEROBOT_DIR}/.venv"
  echo "  Run: (cd ${LEROBOT_DIR} && uv sync)"
  exit 1
fi

echo "==> convert RMBench ${TASK_SET} → ${OUTPUT_DIR} (mode=video, n=${NUM}, setting=${SETTING})"
cd "${LEROBOT_DIR}"
EXTRA=()
if [[ "${OVERWRITE:-0}" != "0" ]]; then
  EXTRA+=(--overwrite)
fi
uv run python "${SCRIPT_DIR}/convert_rmbench_data_to_lerobot.py" "${NUM}" \
  --task-set "${TASK_SET}" \
  --setting "${SETTING}" \
  --rmbench-root "${RMBENCH_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --repo-id "${OUT_NAME}" \
  --mode video \
  "${EXTRA[@]}"

echo "All done."
