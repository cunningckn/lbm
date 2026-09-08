#!/usr/bin/env bash
# Build {dump}/.cache/episodes for every on-disk dump.
# Edit DATASETS below (or DATASET=kai0,libero).
#
#   ./scripts/build_scan_index.sh
#   RESCAN=1 ./scripts/build_scan_index.sh
#   DATASET=kai0 ./scripts/build_scan_index.sh
#   DATASET=kai0,libero ./scripts/build_scan_index.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

DATASETS=(
  abc
  agibot
  das_gripper
  droid
  egoverse
  galaxea
  hifi_umi
  hy_lance
  kai0
  libero
  rmbench
  robotwin
)

DATASET="${DATASET:-$(IFS=,; echo "${DATASETS[*]}")}"
DATA_ROOT="${DATA_ROOT:-}"
RESCAN="${RESCAN:-0}"

exec uv run python "${ROOT}/scripts/build_scan_index.py" \
  --dataset "${DATASET}" \
  ${DATA_ROOT:+--data-root "${DATA_ROOT}"} \
  $([[ "${RESCAN}" != "0" ]] && echo --rescan || true) \
  "$@"
