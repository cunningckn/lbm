#!/usr/bin/env bash
# Extract RMBench language_annotation.json for all tasks into LeRobot sidecars.
#
# Usage (from openpi root):
#   bash examples/rmbench/extract_language_annotations.sh
#   bash examples/rmbench/extract_language_annotations.sh ./data/rmbench_m1 m1 50
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEROBOT_ROOT="${1:-${OPENPI_ROOT}/data/rmbench}"
TASK_SET="${2:-all}"
EPISODES="${3:-50}"

cd "${OPENPI_ROOT}"
exec uv run python examples/rmbench/extract_language_annotations.py \
  --lerobot-root "${LEROBOT_ROOT}" \
  --task-set "${TASK_SET}" \
  --episodes-per-task "${EPISODES}"
