#!/usr/bin/env bash
# Build {dump}/.cache/episodes for every on-disk dump.
# Defaults to all registered datasets (or DATASET=kai0,libero).
#
#   ./scripts/build_scan_index.sh
#   RESCAN=1 ./scripts/build_scan_index.sh
#   DATASET=kai0 ./scripts/build_scan_index.sh
#   DATASET=kai0,libero ./scripts/build_scan_index.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

DATASET="${DATASET:-}"
DATA_ROOT="${DATA_ROOT:-}"
RESCAN="${RESCAN:-0}"

exec uv run python "${ROOT}/scripts/build_scan_index.py" \
  --dataset "${DATASET}" \
  ${DATA_ROOT:+--data-root "${DATA_ROOT}"} \
  $([[ "${RESCAN}" != "0" ]] && echo --rescan || true) \
  "$@"
