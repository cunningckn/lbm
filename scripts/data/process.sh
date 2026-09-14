#!/usr/bin/env bash
# Official download → LBM dump layout (not scan / FK / mmap / norm).
# Edit DATASETS below (or DATASET=galaxea).
#
#   ./scripts/data/process.sh
#   DATASET=galaxea ./scripts/data/process.sh
#   DATASET=kai0,libero ./scripts/data/process.sh
#   FORCE=1 DATASET=rmbench ./scripts/data/process.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/scripts/data${PYTHONPATH:+:${PYTHONPATH}}"
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
  interndata_a1
  kai0
  libero
  molmoact
  rmbench
  robocoin
  robotwin
)

DATASET="${DATASET:-$(IFS=,; echo "${DATASETS[*]}")}"
FORCE="${FORCE:-0}"

exec uv run python "${ROOT}/scripts/data/process.py" \
  --dataset "${DATASET}" \
  $([[ "${FORCE}" != "0" ]] && echo --force || true) \
  "$@"
