#!/usr/bin/env bash
# Train LBM on a LeRobot dataset or mixture.
#
#   DATASET=kai0 ./scripts/train.sh
#   DATASET=rmbench ROBOT_TYPE=rmbench ./scripts/train.sh
#   DATA_MIX=all ./scripts/train.sh
#   NPROC=8 ./scripts/train.sh --fsdp
#   DUMP_BATCH=0 DATASET=kai0 ./scripts/train.sh   # skip first-batch dump
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

DATASET="${DATASET:-}"
ROBOT_TYPE="${ROBOT_TYPE:-}"
DATA_MIX="${DATA_MIX:-}"
DATA_ROOT="${DATA_ROOT:-}"
NPROC="${NPROC:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints}"
ACTION_LENGTH="${ACTION_LENGTH:-}"
ACTION_FREQ="${ACTION_FREQ:-}"
HISTORY_LENGTH="${HISTORY_LENGTH:-}"
HISTORY_FREQ="${HISTORY_FREQ:-}"
DUMP_BATCH="${DUMP_BATCH:-1}"

CMD=(uv run python "${ROOT}/scripts/train.py")
if [[ "${NPROC}" -gt 1 ]]; then
  CMD=(uv run torchrun --standalone --nproc_per_node="${NPROC}" "${ROOT}/scripts/train.py")
fi

exec "${CMD[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  ${DATASET:+--dataset "${DATASET}"} \
  ${ROBOT_TYPE:+--robot-type "${ROBOT_TYPE}"} \
  ${DATA_MIX:+--data-mix "${DATA_MIX}"} \
  ${DATA_ROOT:+--data-root "${DATA_ROOT}"} \
  ${ACTION_LENGTH:+--action-length "${ACTION_LENGTH}"} \
  ${ACTION_FREQ:+--action-freq "${ACTION_FREQ}"} \
  ${HISTORY_LENGTH:+--history-length "${HISTORY_LENGTH}"} \
  ${HISTORY_FREQ:+--history-freq "${HISTORY_FREQ}"} \
  $([[ "${DUMP_BATCH}" != "0" ]] && echo --dump-batch || echo --no-dump-batch) \
  "$@"
