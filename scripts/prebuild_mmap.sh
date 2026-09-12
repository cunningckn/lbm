#!/usr/bin/env bash
# Prebuild parquet trajectory mmap + JPEG frame mmap caches.
# Defaults to all registered datasets (or DATASET=kai0,libero).
#
#   ./scripts/prebuild_mmap.sh
#   DATASET=libero ./scripts/prebuild_mmap.sh
#   DATASET=kai0,libero WORKERS=16 ./scripts/prebuild_mmap.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

DATASET="${DATASET:-}"
DATA_ROOT="${DATA_ROOT:-}"
WORKERS="${WORKERS:-0}"

exec uv run python "${ROOT}/scripts/prebuild_mmap.py" \
  --workers "${WORKERS}" \
  --dataset "${DATASET}" \
  ${DATA_ROOT:+--data-root "${DATA_ROOT}"} \
  "$@"
