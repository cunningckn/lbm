#!/bin/bash
# Terminal 2 — LIBERO sim client (simulation/libero/.venv).
# Requires the LBM policy server from eval_policy.sh.
#
# Usage:
#   bash simulation/libero/eval_env.sh
#   bash simulation/libero/eval_env.sh --task-suite-name libero_spatial

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

EXAMPLE_DIR="${SCRIPT_DIR}"
VENV_DIR="${EXAMPLE_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

cd "${LBM_ROOT}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: missing LIBERO venv at $VENV_DIR"
  echo "  Run: bash simulation/libero/install_env.sh"
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
export LIBERO_ROOT

_local_hosts="127.0.0.1,localhost,${HOST}"
export no_proxy="${no_proxy:+$no_proxy,}${_local_hosts}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${_local_hosts}"

echo "==> LIBERO eval client"
echo "    venv:   $VENV_DIR"
echo "    suite:  $TASK_SUITE"
echo "    server: $HOST:$PORT"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$VENV_PYTHON" "$EXAMPLE_DIR/main.py" \
  --host "$HOST" \
  --port "$PORT" \
  --task-suite-name "$TASK_SUITE" \
  "$@"
