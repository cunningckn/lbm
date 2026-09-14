#!/usr/bin/env bash
# Short matched history validation, sequential on one GPU. Outputs remain private.
set -euo pipefail
OUTPUT_ROOT="${1:?provide a new output root}"
LBM_TEST_PYTHON="${LBM_PYTHON:-python}"
mkdir "$OUTPUT_ROOT"
for SEED in 123 456; do
  for MODE in current vision state both; do
    "$LBM_TEST_PYTHON" context/validation/history/run_matched.py \
      --mode "$MODE" --seed "$SEED" --batch 64 --steps 100 --val-batches 4 \
      --output "$OUTPUT_ROOT/$SEED-$MODE" > "$OUTPUT_ROOT/$SEED-$MODE.log" 2>&1
  done
done
