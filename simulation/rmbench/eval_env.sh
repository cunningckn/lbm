#!/bin/bash
# Terminal 2 — RMBench sim client (simulation/rmbench/.venv).
# Requires the LBM policy server from eval_policy.sh.
#
# Usage:
#   bash simulation/rmbench/eval_env.sh
#   bash simulation/rmbench/eval_env.sh put_back_block

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

EXAMPLE_DIR="${SCRIPT_DIR}"
VENV_DIR="${EXAMPLE_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

cd "${LBM_ROOT}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: missing RMBench venv at $VENV_DIR"
  echo "  Run: bash simulation/rmbench/install_env.sh"
  exit 1
fi

if [[ ! -d "$RMBENCH_ROOT/envs" ]]; then
  echo "ERROR: RMBench not found at $RMBENCH_ROOT"
  exit 1
fi

TASK_NAME="${TASK_NAME:-put_back_block}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
NUM_TRIALS="${NUM_TRIALS:-50}"

if [[ $# -ge 1 && ! "$1" =~ ^- ]]; then
  TASK_NAME="$1"
  shift
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$RMBENCH_ROOT:${PYTHONPATH:-}"
export RMBENCH_ROOT

_local_hosts="127.0.0.1,localhost,${HOST}"
export no_proxy="${no_proxy:+$no_proxy,}${_local_hosts}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${_local_hosts}"

echo "==> RMBench eval client"
echo "    venv:   $VENV_DIR"
echo "    task:   $TASK_NAME"
echo "    server: $HOST:$PORT"
echo "    trials: $NUM_TRIALS"

_RMBENCH_VK="$EXAMPLE_DIR/lib/vulkan"
if [[ -f "$_RMBENCH_VK/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="$_RMBENCH_VK/nvidia_icd.json"
fi
if [[ -f "$_RMBENCH_VK/libvulkan.so.1" ]]; then
  export LD_LIBRARY_PATH="$_RMBENCH_VK:${LD_LIBRARY_PATH:-}"
fi
export MUJOCO_GL=egl

"$VENV_PYTHON" "$EXAMPLE_DIR/main.py" \
  --task-name "$TASK_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --num-trials "$NUM_TRIALS" \
  --rmbench-root "$RMBENCH_ROOT" \
  "$@"
