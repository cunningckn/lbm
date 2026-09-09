# Shared download helpers. Sourced by download.sh / download_<name>.sh.
# HF_ENDPOINT defaults to the mirror; override to https://huggingface.co if needed.

_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${_DATA_DIR}/../.." && pwd)"
DATASETS_DIR="${LBM_DATASETS:-${ROOT}/datasets}"
RAW_DIR="${LBM_DATA_RAW:-${DATASETS_DIR}/raw}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HFD="${_DATA_DIR}/hfd.sh"
CATALOG_PY="${_DATA_DIR}/catalog.py"

load_catalog() {
  local name="$1"
  eval "$(python3 "${CATALOG_PY}" --bash "${name}")"
}

print_urls() {
  local u
  printf 'dump: %s\n' "${DUMP_NAME}"
  for u in "${DUMP_URLS[@]}"; do
    printf 'url:  %s\n' "${u}"
  done
  if [[ -n "${PROCESS_NOTE}" ]]; then
    printf 'process: %s (%s)\n' "${PROCESS}" "${PROCESS_NOTE}"
  else
    printf 'process: %s\n' "${PROCESS}"
  fi
}

raw_dest() {
  local name="$1"
  local subdir="${2:-}"
  if [[ -n "${subdir}" ]]; then
    printf '%s/%s/%s' "${RAW_DIR}" "${name}" "${subdir}"
  else
    printf '%s/%s' "${RAW_DIR}" "${name}"
  fi
}

dump_dest() {
  printf '%s/%s' "${DATASETS_DIR}" "$1"
}

# Download one HF repo into dest. Extra args are forwarded to hfd.sh.
hfd_dataset() {
  local repo="$1"
  local dest="$2"
  shift 2
  mkdir -p "${dest}"
  printf 'hfd:  https://huggingface.co/datasets/%s\n' "${repo}"
  printf 'dest: %s\n' "${dest}"
  bash "${HFD}" "${repo}" --dataset --local-dir "${dest}" "$@"
}

link_if_local() {
  local name="$1"
  local dest
  dest="$(dump_dest "${name}")"
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
}
