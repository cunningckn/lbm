#!/usr/bin/env bash
# Download the official corpus for one mix dump into datasets/raw/<name>.
#   ./scripts/data/download.sh droid
#   ./scripts/data/download_droid.sh
#   ./scripts/data/download.sh galaxea --link-local   # cluster shortcut
#
# Default is a real Hugging Face (or EgoVerse S3) fetch. URLs are printed
# before any download. --link-local only symlinks a processed dump if present.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

name="${1:-}"
if [[ -z "${name}" ]]; then
  printf 'usage: %s <dump-name> [--link-local] [hfd args...]\n' "$(basename "$0")" >&2
  python3 "${CATALOG_PY}" --list >&2
  exit 1
fi
shift

LINK_LOCAL=0
HFD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --link-local) LINK_LOCAL=1; shift ;;
    *) HFD_ARGS+=("$1"); shift ;;
  esac
done

load_catalog "${name}"
print_urls

if [[ "${LINK_LOCAL}" -eq 1 ]]; then
  link_if_local "${name}"
  exit $?
fi

if [[ ${#HF_REPO[@]} -eq 0 ]]; then
  if [[ -n "${S3_CMD}" ]]; then
    dest="$(raw_dest "${name}")"
    mkdir -p "${dest}"
    printf 'EgoVerse is not on Hugging Face. From a clone of\n  %s\nrun:\n  %s\n' "${DUMP_URL}" "${S3_CMD/DEST/${dest}}"
    printf 'Requires their AWS/R2 credentials (do not copy keys into this repo).\n'
    printf 'raw dest: %s\n' "${dest}"
    exit 0
  fi
  printf 'no download URL for %s\n' "${name}" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}"
i=0
for repo in "${HF_REPO[@]}"; do
  subdir="${HF_SUBDIR[$i]:-}"
  include="${HF_INCLUDE[$i]:-}"
  dest="$(raw_dest "${name}" "${subdir}")"
  extra=()
  if [[ -n "${include}" ]]; then
    extra+=(--include "${include}")
  fi
  if [[ -e "${dest}" && -z "${LBM_DOWNLOAD_FORCE:-}" ]]; then
    printf 'already present: %s\n' "${dest}"
  else
    hfd_dataset "${repo}" "${dest}" "${extra[@]}" "${HFD_ARGS[@]}"
  fi
  i=$((i + 1))
done
printf 'raw: %s\n' "$(raw_dest "${name}")"
printf 'next: ./scripts/data/process.sh %s\n' "${name}"
