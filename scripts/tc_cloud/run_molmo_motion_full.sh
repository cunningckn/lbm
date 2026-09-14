#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

SOURCE_ROOT=${SOURCE_ROOT:-/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1}
MODE=${MODE:-full}
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ "$MODE" != "full" && "$MODE" != "smoke" ]]; then
  echo "MODE must be full or smoke, got: $MODE" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "source root does not exist: $SOURCE_ROOT" >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT_ROOT=${SMOKE_OUTPUT:-"/tmp/molmo-motion-cache-smoke-${USER:-user}-$$"}
  WORKERS=${WORKERS:-8}
  SHARD_SIZE=${SHARD_SIZE:-2}
  CHECKS=${CHECKS:-2}
  VERIFY_SOURCE_HASHES=0
  CHECKSUMS=0
else
  WORKERS=${WORKERS:-100}
  SHARD_SIZE=${SHARD_SIZE:-256}
  CHECKS=${CHECKS:-32}
  VERIFY_SOURCE_HASHES=${VERIFY_SOURCE_HASHES:-1}
  CHECKSUMS=${CHECKSUMS:-1}
fi

mkdir -p -- "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.build.lock"
if ! flock -n 9; then
  echo "another MolmoMotion build holds $OUTPUT_ROOT/.build.lock" >&2
  exit 3
fi

export PYTHONPATH="$REPO_ROOT/tools/molmo_motion_cache/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

args=(
  -m molmo_motion_cache build-release
  --source-root "$SOURCE_ROOT"
  --output "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --shard-size "$SHARD_SIZE"
  --checks "$CHECKS"
)
if [[ "$MODE" == "smoke" ]]; then
  args+=(--limit-per-subset 1 --no-checksums)
elif [[ "$VERIFY_SOURCE_HASHES" == "1" ]]; then
  args+=(--verify-source-hashes)
fi
if [[ "$MODE" == "full" && "$CHECKSUMS" != "1" ]]; then
  args+=(--no-checksums)
fi

echo "repo_root=$REPO_ROOT"
echo "source_root=$SOURCE_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "mode=$MODE workers=$WORKERS shard_size=$SHARD_SIZE checks=$CHECKS"
echo "verify_source_hashes=$VERIFY_SOURCE_HASHES checksums=$CHECKSUMS"
"$PYTHON_BIN" -c 'import numpy, pyarrow; print("dependencies: numpy=" + numpy.__version__ + " pyarrow=" + pyarrow.__version__)'
"$PYTHON_BIN" "${args[@]}"
