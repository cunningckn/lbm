#!/bin/bash
# Terminal 1 — LBM policy HTTP server (repo-root uv env).
#
# Usage:
#   bash simulation/rmbench/eval_policy.sh ./checkpoints/<run>/<step>.pt
#   CKPT=./checkpoints/<run> PORT=8000 bash simulation/rmbench/eval_policy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

cd "${LBM_ROOT}"

CKPT="${CKPT:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
ROBOT_TYPE="${ROBOT_TYPE:-rmbench}"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --robot-type) ROBOT_TYPE="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --port=*|--host=*|--robot-type=*|--ckpt=*)
      key="${1%%=*}"
      val="${1#*=}"
      case "$key" in
        --port) PORT="$val" ;;
        --host) HOST="$val" ;;
        --robot-type) ROBOT_TYPE="$val" ;;
        --ckpt) CKPT="$val" ;;
      esac
      shift
      ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

if [[ $# -ge 1 && ! "$1" =~ ^- ]]; then
  CKPT="$1"
  shift
fi

if [[ -z "${CKPT}" ]]; then
  echo "ERROR: pass a checkpoint path"
  echo "  bash simulation/rmbench/eval_policy.sh ./checkpoints/<run>/<step>.pt"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "==> Serving LBM RMBench policy"
echo "    ckpt:  $CKPT"
echo "    type:  $ROBOT_TYPE"
echo "    bind:  $HOST:$PORT"

uv run python scripts/serve_policy.py \
  --ckpt "$CKPT" \
  --robot-type "$ROBOT_TYPE" \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
