#!/usr/bin/env bash
# Prepare one dataset and build its mmap cache on tc_dev.
#
# Examples:
#   DATASET=droid CHECK_ONLY=1 ./scripts/data/convert_50cpu.sh
#   DATASET=interndata_a1 SOURCE=/path/to/InternData-A1 ./scripts/data/convert_50cpu.sh
#   DATASET=molmoact BUILD_MMAP=0 ./scripts/data/convert_50cpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${DATASET:-}"
DATA_BASE="${DATA_BASE:-/home/tione/workspace/kainingchen/Datasets}"
LBM_DATASETS="${LBM_DATASETS:-${DATA_BASE}/lbm}"
AVAILABLE_CPUS="$(nproc)"
DEFAULT_WORKERS="${AVAILABLE_CPUS}"
if (( DEFAULT_WORKERS > 48 )); then
  DEFAULT_WORKERS=48
fi
WORKERS="${WORKERS:-${DEFAULT_WORKERS}}"
BUILD_MMAP="${BUILD_MMAP:-1}"
FORCE="${FORCE:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

if [[ -z "${DATASET}" || "${DATASET}" == *,* ]]; then
  printf 'DATASET must contain exactly one catalog name\n' >&2
  exit 2
fi
if [[ ! "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'WORKERS must be a positive integer, got %s\n' "${WORKERS}" >&2
  exit 2
fi

case "${DATASET}" in
  droid) default_source="${DATA_BASE}/Droid/droid_1.0.1" ;;
  interndata_a1) default_source="${DATA_BASE}/InternData-A1" ;;
  molmoact) default_source="${DATA_BASE}/MolmoAct-Dataset" ;;
  robocoin) default_source="${DATA_BASE}/RoboCOIN" ;;
  *) default_source="${DATA_BASE}/${DATASET}" ;;
esac
SOURCE="${SOURCE:-${default_source}}"
DEST="${LBM_DATASETS}/${DATASET}"

if [[ ! -d "${SOURCE}" ]]; then
  printf 'source dataset is missing: %s\n' "${SOURCE}" >&2
  printf 'set SOURCE=/absolute/path/to/the/downloaded-dataset\n' >&2
  exit 1
fi
if ! python3 "${ROOT}/scripts/data/catalog.py" --list | grep -Fxq "${DATASET}"; then
  printf 'unknown dataset: %s\n' "${DATASET}" >&2
  exit 2
fi
if (( WORKERS > AVAILABLE_CPUS )); then
  printf 'WORKERS=%s exceeds CPUs available to this job (%s)\n' "${WORKERS}" "${AVAILABLE_CPUS}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'PYTHON_BIN is not executable: %s\n' "${PYTHON_BIN}" >&2
    exit 1
  fi
  python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${python_version}" != "3.12" ]]; then
    printf 'PYTHON_BIN must use Python 3.12, got %s\n' "${python_version}" >&2
    exit 1
  fi
  python_runner=("${PYTHON_BIN}")
  runner_label="${PYTHON_BIN}"
else
  UV_BIN="${UV_BIN:-$(command -v uv || true)}"
  if [[ -z "${UV_BIN}" && -x /root/.local/bin/uv ]]; then
    UV_BIN=/root/.local/bin/uv
  fi
  if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
    printf 'uv was not found; set UV_BIN or a prepared PYTHON_BIN\n' >&2
    exit 1
  fi
  python_runner=("${UV_BIN}" run --python 3.12 --extra data python)
  runner_label="${UV_BIN} run --python 3.12 --extra data"
fi

printf 'project:   %s\n' "${ROOT}"
printf 'dataset:   %s\n' "${DATASET}"
printf 'source:    %s\n' "${SOURCE}"
printf 'dest:      %s\n' "${DEST}"
printf 'workers:   %s\n' "${WORKERS}"
printf 'runner:    %s\n' "${runner_label}"
printf 'build mmap: %s\n' "${BUILD_MMAP}"

if [[ "${CHECK_ONLY}" != "0" ]]; then
  printf 'configuration check passed\n'
  exit 0
fi

mkdir -p "${LBM_DATASETS}"
export LBM_DATASETS
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts/data${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "${ROOT}"

force_args=()
if [[ "${FORCE}" != "0" ]]; then
  force_args+=(--force)
fi

"${python_runner[@]}" "${ROOT}/scripts/data/process.py" \
  --dataset "${DATASET}" \
  --raw "${SOURCE}" \
  --dest "${DEST}" \
  "${force_args[@]}"

if [[ "${BUILD_MMAP}" != "0" ]]; then
  "${python_runner[@]}" "${ROOT}/scripts/prebuild_mmap.py" \
    --dataset "${DATASET}" \
    --data-root "${LBM_DATASETS}" \
    --workers "${WORKERS}"
fi
