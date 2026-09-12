#!/usr/bin/env bash
# Read dumps through the training DataLoader, loop, dump the first batch.
# Defaults to all registered datasets (or DATASET=kai0,libero).
#
#   ./scripts/inspect_data.sh
#   DATASET=rmbench ./scripts/inspect_data.sh
#   MAX_BATCHES=1 DATASET=kai0 ./scripts/inspect_data.sh
#   MAX_BATCHES=8 HISTORY_LENGTH=1 DATASET=rmbench ./scripts/inspect_data.sh
#   DATASET=kai0,libero ./scripts/inspect_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

DATASET="${DATASET:-}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/inspect_data}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCHES="${MAX_BATCHES:-}"
ACTION_LENGTH="${ACTION_LENGTH:-}"
ACTION_FREQ="${ACTION_FREQ:-}"
HISTORY_LENGTH="${HISTORY_LENGTH:-}"
HISTORY_FREQ="${HISTORY_FREQ:-}"

exec uv run python "${ROOT}/scripts/inspect_data.py" \
  --output-dir "${OUTPUT_DIR}" \
  --num-workers "${NUM_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --dataset "${DATASET}" \
  ${DATA_ROOT:+--data-root "${DATA_ROOT}"} \
  ${MAX_BATCHES:+--max-batches "${MAX_BATCHES}"} \
  ${ACTION_LENGTH:+--action-length "${ACTION_LENGTH}"} \
  ${ACTION_FREQ:+--action-freq "${ACTION_FREQ}"} \
  ${HISTORY_LENGTH:+--history-length "${HISTORY_LENGTH}"} \
  ${HISTORY_FREQ:+--history-freq "${HISTORY_FREQ}"} \
  "$@"
