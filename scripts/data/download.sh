#!/usr/bin/env bash
# Download official corpora into datasets/raw/<name>. Edit DATASETS below
# (or DATASET=droid). URLs live in catalog.py.
#
#   ./scripts/data/download.sh
#   DATASET=droid ./scripts/data/download.sh
#   DATASET=kai0,libero ./scripts/data/download.sh
#   LINK_LOCAL=1 DATASET=droid ./scripts/data/download.sh
#   FORCE=1 DATASET=galaxea ./scripts/data/download.sh
set -euo pipefail

_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${_DATA_DIR}/../.." && pwd)"
DATASETS_DIR="${LBM_DATASETS:-${ROOT}/datasets}"
RAW_DIR="${LBM_DATA_RAW:-${DATASETS_DIR}/raw}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HFD="${_DATA_DIR}/hfd.sh"
CATALOG_PY="${_DATA_DIR}/catalog.py"

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
LINK_LOCAL="${LINK_LOCAL:-0}"
FORCE="${FORCE:-${LBM_DOWNLOAD_FORCE:-0}}"

load_catalog() {
  eval "$(python3 "${CATALOG_PY}" --bash "$1")"
}

download_one() {
  local name="$1"
  shift
  load_catalog "${name}"
  printf '== %s ==\n' "${name}"
  local u
  for u in "${DUMP_URLS[@]}"; do
    printf 'url:  %s\n' "${u}"
  done
  printf 'process: %s (%s)\n' "${PROCESS}" "${PROCESS_NOTE}"

  if [[ "${LINK_LOCAL}" != "0" ]]; then
    local dest="${DATASETS_DIR}/${name}"
    if [[ -z "${LOCAL_DUMP}" || ! -e "${LOCAL_DUMP}" ]]; then
      printf 'no local dump at %s\n' "${LOCAL_DUMP:-?}" >&2
      return 1
    fi
    mkdir -p "${DATASETS_DIR}"
    if [[ -e "${dest}" || -L "${dest}" ]]; then
      printf 'already present: %s\n' "${dest}"
      return 0
    fi
    ln -s "${LOCAL_DUMP}" "${dest}"
    printf 'linked %s -> %s\n' "${dest}" "${LOCAL_DUMP}"
    return 0
  fi

  if [[ ${#HF_REPO[@]} -eq 0 ]]; then
    if [[ -n "${S3_CMD}" ]]; then
      local dest="${RAW_DIR}/${name}"
      mkdir -p "${dest}"
      printf 'EgoVerse is not on Hugging Face. From a clone of\n  %s\nrun:\n  %s\n' \
        "${DUMP_URL}" "${S3_CMD/DEST/${dest}}"
      printf 'Requires their AWS/R2 credentials (do not copy keys into this repo).\n'
      printf 'raw dest: %s\n' "${dest}"
      return 0
    fi
    printf 'no download URL for %s\n' "${name}" >&2
    return 1
  fi

  mkdir -p "${RAW_DIR}"
  local i=0 repo subdir include dest extra
  for repo in "${HF_REPO[@]}"; do
    subdir="${HF_SUBDIR[$i]:-}"
    include="${HF_INCLUDE[$i]:-}"
    if [[ -n "${subdir}" ]]; then
      dest="${RAW_DIR}/${name}/${subdir}"
    else
      dest="${RAW_DIR}/${name}"
    fi
    extra=()
    if [[ -n "${include}" ]]; then
      extra+=(--include "${include}")
    fi
    if [[ -e "${dest}" && "${FORCE}" == "0" ]]; then
      printf 'already present: %s\n' "${dest}"
    else
      mkdir -p "${dest}"
      printf 'hfd:  https://huggingface.co/datasets/%s\n' "${repo}"
      printf 'dest: %s\n' "${dest}"
      bash "${HFD}" "${repo}" --dataset --local-dir "${dest}" "${extra[@]}" "$@"
    fi
    i=$((i + 1))
  done
  printf 'raw: %s\n' "${RAW_DIR}/${name}"
  printf 'next: DATASET=%s ./scripts/data/process.sh\n' "${name}"
}

IFS=',' read -r -a NAMES <<< "${DATASET}"
for name in "${NAMES[@]}"; do
  name="${name// /}"
  [[ -z "${name}" ]] && continue
  download_one "${name}" "$@"
done
