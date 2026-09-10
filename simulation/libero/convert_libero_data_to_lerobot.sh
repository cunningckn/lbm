#!/bin/bash
# Convert LIBERO RLDS → LeRobot under lbm/datasets/libero.
#
# Usage:
#   bash simulation/libero/convert_libero_data_to_lerobot.sh /path/to/modified_libero_rlds
#   LIBERO_RLDS=/path/to/rlds bash simulation/libero/convert_libero_data_to_lerobot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

DATA_DIR="${1:-${LIBERO_RLDS:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASETS}/libero}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 /path/to/modified_libero_rlds" >&2
  echo "  or set LIBERO_RLDS" >&2
  exit 1
fi

if [[ ! -x "${LEROBOT_DIR}/.venv/bin/python" ]]; then
  echo "ERROR: missing convert env at ${LEROBOT_DIR}/.venv"
  echo "  Run: (cd ${LEROBOT_DIR} && uv sync)"
  exit 1
fi

echo "==> convert LIBERO RLDS ${DATA_DIR} → ${OUTPUT_DIR}"
cd "${LEROBOT_DIR}"
EXTRA=()
if [[ "${OVERWRITE:-0}" != "0" ]]; then
  EXTRA+=(--overwrite)
fi
uv run python "${SCRIPT_DIR}/convert_libero_data_to_lerobot.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --repo-id libero \
  "${EXTRA[@]}"

echo "All done."
