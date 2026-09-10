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
VENV_DIR="${EXAMPLE_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -d "${LIBERO_ROOT}" ]]; then
  echo "LIBERO not found at ${LIBERO_ROOT}"
  exit 1
fi

cd "${LBM_ROOT}"

echo "==> Creating venv at ${VENV_DIR} (Python 3.10)"
rm -rf "${VENV_DIR}"
# System 3.10 often lacks python3.10-dev (Python.h). robosuite → pynput → evdev
# builds a C extension against it, so use uv-managed CPython which ships headers.
uv python install 3.10
uv venv --python 3.10 --managed-python --no-project "${VENV_DIR}"

echo "==> Installing LIBERO client requirements"
# Keyboard teleop only; OffScreenRenderEnv eval does not import these.
EXCLUDES_FILE="$(mktemp)"
trap 'rm -f "${EXCLUDES_FILE}"' EXIT
printf '%s\n' pynput evdev python-xlib > "${EXCLUDES_FILE}"

uv pip install -p "${VENV_PYTHON}" -r "${EXAMPLE_DIR}/requirements.in" \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy unsafe-best-match \
  --excludes "${EXCLUDES_FILE}" || \
uv pip install -p "${VENV_PYTHON}" \
  --excludes "${EXCLUDES_FILE}" \
  "imageio[ffmpeg]" numpy tqdm tyro PyYAML opencv-python matplotlib robosuite==1.4.1

echo "==> Installing third_party/libero (editable)"
uv pip install -p "${VENV_PYTHON}" -e "${LIBERO_ROOT}"

cat <<EOF

Environment ready.

  source ${EXAMPLE_DIR}/.venv/bin/activate
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
