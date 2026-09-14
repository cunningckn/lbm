#!/bin/bash
# LIBERO eval client venv (sim only). Data convert uses simulation/lerobot.
# The policy is served from the LBM root uv env (eval_policy.sh).
#
# Usage:
#   bash simulation/libero/install_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

EXAMPLE_DIR="${SCRIPT_DIR}"
VENV_DIR="${LIBERO_VENV}"
VENV_PYTHON="${VENV_DIR}/bin/python"
UV_BIN="${UV_BIN:-uv}"

if [[ ! -d "${LIBERO_ROOT}" ]]; then
  echo "LIBERO not found at ${LIBERO_ROOT}"
  exit 1
fi

cd "${LBM_ROOT}"

echo "==> Creating venv at ${VENV_DIR} (Python 3.10)"
# Reuse an available 3.10 interpreter; uv can download one if none is available.
LIBERO_PYTHON="${LIBERO_PYTHON:-3.10}"
"$UV_BIN" venv --python "$LIBERO_PYTHON" --allow-existing --no-project "${VENV_DIR}"
"$VENV_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else "LIBERO requires Python 3.10")'

echo "==> Installing LIBERO client requirements"
"$UV_BIN" pip install -p "${VENV_PYTHON}" -r "${EXAMPLE_DIR}/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match \
  --excludes "${EXAMPLE_DIR}/headless-excludes.txt"

echo "==> Installing third_party/libero (editable)"
"$UV_BIN" pip install -p "${VENV_PYTHON}" --no-deps --no-build-isolation -e "${LIBERO_ROOT}"

cat <<EOF

Environment ready.

  export LIBERO_VENV="${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  export PYTHONPATH=${LIBERO_ROOT}:\$PYTHONPATH
  export LIBERO_ROOT=${LIBERO_ROOT}

Data prep (RLDS → LeRobot, simulation/lerobot uv):
  bash simulation/libero/convert_libero_data_to_lerobot.sh /path/to/modified_libero_rlds

Train (lbm root uv):
  DATASET=libero ./scripts/train.sh

Eval (two terminals):
  bash simulation/libero/eval_policy.sh ./checkpoints/<run>/<step>.pt
  bash simulation/libero/eval_env.sh --task-suite-name libero_spatial

EOF
