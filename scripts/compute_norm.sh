#!/usr/bin/env bash
# Compute state/action mean/std/min/max/q01/q99 next to each dump.
# Defaults to all registered datasets (or DATASET=kai0,libero). Set MMAP=1 to also prebuild JPEG mmap.
#
#   ./scripts/compute_norm.sh
#   DATASET=libero ./scripts/compute_norm.sh
#   DATASET=kai0,libero ./scripts/compute_norm.sh
#   ACTION_KIND=eef ACTION_FREQ=10 ACTION_LENGTH=5 DATASET=kai0 ./scripts/compute_norm.sh
#   MMAP=1 WORKERS=16 DATASET=libero ./scripts/compute_norm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

DATASET="${DATASET:-}"
DATA_ROOT="${DATA_ROOT:-}"
WORKERS="${WORKERS:-0}"
MMAP="${MMAP:-0}"

ACTION_FREQ="${ACTION_FREQ:-}"
ACTION_LENGTH="${ACTION_LENGTH:-}"
ACTION_MODE="${ACTION_MODE:-}"
ACTION_KIND="${ACTION_KIND:-}"
ACTION_FORMAT="${ACTION_FORMAT:-}"

exec uv run python "${ROOT}/scripts/compute_norm.py" \
  --workers "${WORKERS}" \
  --dataset "${DATASET}" \
  ${DATA_ROOT:+--data-root "${DATA_ROOT}"} \
  ${ACTION_FREQ:+--action-freq "${ACTION_FREQ}"} \
  ${ACTION_LENGTH:+--action-length "${ACTION_LENGTH}"} \
  ${ACTION_MODE:+--action-mode "${ACTION_MODE}"} \
  ${ACTION_KIND:+--action-kind "${ACTION_KIND}"} \
  ${ACTION_FORMAT:+--action-format "${ACTION_FORMAT}"} \
  $([[ "${MMAP}" != "0" ]] && echo --mmap || true) \
  "$@"
