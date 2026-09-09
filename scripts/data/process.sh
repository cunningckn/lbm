#!/usr/bin/env bash
# Official download → LBM dump layout (not scan / FK / mmap / norm).
#   ./scripts/data/process.sh galaxea
#   ./scripts/data/process.sh --all
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/scripts/data${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
if [[ "${1:-}" == "--all" ]]; then
  shift
  exec uv run python "${ROOT}/scripts/data/process.py" --all "$@"
fi
if [[ "${1:-}" == --* || -z "${1:-}" ]]; then
  exec uv run python "${ROOT}/scripts/data/process.py" "$@"
fi
name="$1"
shift
exec uv run python "${ROOT}/scripts/data/process.py" --dataset "${name}" "$@"
