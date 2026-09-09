#!/usr/bin/env bash
# Download every mix dump (official URLs). Pass --link-local to symlink cluster dumps.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mapfile -t NAMES < <(python3 "${DIR}/catalog.py" --list)
for name in "${NAMES[@]}"; do
  printf '== %s ==\n' "${name}"
  bash "${DIR}/download.sh" "${name}" "$@"
done
