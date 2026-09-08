#!/usr/bin/env bash
# Precompute joint→EEF proprio into {dump}/.cache/fk.
# Edit DATASETS below (or DATASET=kai0,agibot). Dumps without an FK chain are skipped.
#
#   ./scripts/prebuild_fk.sh
#   DATASET=kai0 ./scripts/prebuild_fk.sh
#   DATASET=kai0,agibot WORKERS=16 ./scripts/prebuild_fk.sh
#   FORCE=1 DATASET=kai0 ./scripts/prebuild_fk.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

DATASETS=(
  abc
  agibot
  # das_gripper
  droid
  # egoverse
  galaxea
  # hifi_umi
  # hy_lance
  kai0
  # libero
  rmbench
  robotwin
)

DATASET="${DATASET:-$(IFS=,; echo "${DATASETS[*]}")}"
DATA_ROOT="${DATA_ROOT:-}"
WORKERS="${WORKERS:-0}"
FORCE="${FORCE:-0}"

exec uv run python "${ROOT}/scripts/prebuild_fk.py" \
  --workers "${WORKERS}" \
  --dataset "${DATASET}" \
  ${DATA_ROOT:+--data-root "${DATA_ROOT}"} \
  $([[ "${FORCE}" != "0" ]] && echo --force || true) \
  "$@"
