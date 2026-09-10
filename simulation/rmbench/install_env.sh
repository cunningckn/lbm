#!/bin/bash
# Create the RMBench eval client / sim venv.
# Data prep uses simulation/lerobot uv (see convert_rmbench_data_to_lerobot.sh).
# The policy itself is served from the LBM root uv env (eval_policy.sh).
#
# Usage:
#   bash simulation/rmbench/install_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "${SCRIPT_DIR}/../env.sh"

EXAMPLE_DIR="${SCRIPT_DIR}"
VENV_DIR="${EXAMPLE_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -d "${RMBENCH_ROOT}" ]]; then
  echo "RMBench not found at ${RMBENCH_ROOT}"
  exit 1
fi

cd "${LBM_ROOT}"

echo "==> Creating venv at $VENV_DIR (Python 3.11)"
rm -rf "$VENV_DIR"
uv venv --python 3.11 "$VENV_DIR"

echo "==> Installing client / eval requirements"
if [[ -f "$EXAMPLE_DIR/requirements.txt" ]]; then
  uv pip sync -p "$VENV_PYTHON" "$EXAMPLE_DIR/requirements.txt"
else
  uv pip install -p "$VENV_PYTHON" -r "$EXAMPLE_DIR/requirements.in"
fi

# sapien still imports pkg_resources (removed in setuptools>=82).
uv pip install -p "$VENV_PYTHON" 'setuptools<82'

echo "==> Installing RMBench sim stack"
CUROBO_DIR="$RMBENCH_ROOT/envs/curobo"
CUROBO_TAG="v0.7.8"
REQS="$RMBENCH_ROOT/script/requirements.txt"
FILTERED_REQS="$(mktemp)"

cleanup() {
  rm -f "$FILTERED_REQS"
}
trap cleanup EXIT

has_working_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1
}

PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

uv_pip_pypi() {
  echo "    \$ uv pip install (PyPI) ..."
  uv pip install -p "$VENV_PYTHON" \
    --index-url "$PYPI_INDEX" \
    "$@"
}

uv_pip_torch() {
  echo "    \$ uv pip install (PyTorch index: $PYTORCH_INDEX) ..."
  uv pip install -p "$VENV_PYTHON" \
    --index-url "$PYTORCH_INDEX" \
    "$@"
}

package_location() {
  local pkg="$1"
  uv pip show -p "$VENV_PYTHON" "$pkg" 2>/dev/null | awk '/^Location: / { print $2 }'
}

grep -v -E '^(torch|torchvision|ffmpeg)([=<>]|$)|^[[:space:]]*#' "$REQS" \
  | awk 'NF && !/^[[:space:]]*#/' > "$FILTERED_REQS"

echo "==> Installing RMBench sim stack into $VENV_DIR"
echo "    Python: $VENV_PYTHON"
echo "    PyPI:   $PYPI_INDEX"
echo "    Torch:  $PYTORCH_INDEX"
if ! has_working_gpu; then
  echo "    GPU:    not detected (using CPU torch wheels)"
fi

echo "==> [1/5] Installing PyTorch (large download; verbose logs follow)"
uv_pip_torch torch==2.4.1 torchvision

echo "==> [2/5] Installing remaining sim requirements"
uv_pip_pypi -r "$FILTERED_REQS"

echo "==> [3/5] Installing pytorch3d (may compile; can take 10+ min)"
# pytorch3d setup.py imports torch at build time; uv build isolation hides it.
uv_pip_pypi --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo "==> [4/5] Patching sapien / mplib"
SAPIEN_LOCATION="$(package_location sapien)/sapien"
URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"
if [[ ! -f "$URDF_LOADER" ]]; then
  echo "sapien urdf_loader not found at $URDF_LOADER"
  exit 1
fi
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"

MPLIB_LOCATION="$(package_location mplib)/mplib"
PLANNER="$MPLIB_LOCATION/planner.py"
if [[ ! -f "$PLANNER" ]]; then
  echo "mplib planner not found at $PLANNER"
  exit 1
fi
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"

echo "==> [5/5] Installing CuRobo (editable)"
if [[ ! -d "$CUROBO_DIR/.git" ]]; then
  git clone --branch "$CUROBO_TAG" --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi
uv_pip_pypi -e "$CUROBO_DIR" --no-build-isolation

# cuRobo v0.7.8 still uses wp.torch.*; removed in warp-lang>=1.13.
uv_pip_pypi 'warp-lang==1.12.1'

# CuRobo ymls bake absolute urdf/collision paths from *_tmp.yml templates.
# Must match this checkout (not a previous absolute path).
echo "==> Rewriting embodiment curobo_*.yml paths -> $RMBENCH_ROOT"
shopt -s nullglob globstar
for tmp in "$RMBENCH_ROOT"/assets/embodiments/**/*_tmp.yml; do
  target="${tmp%_tmp.yml}.yml"
  sed "s|\${ASSETS_PATH}|$RMBENCH_ROOT|g; s|\$ASSETS_PATH|$RMBENCH_ROOT|g" "$tmp" > "$target"
  echo "    ${target#"$RMBENCH_ROOT"/}"
done
shopt -u nullglob globstar

echo "==> Sim stack installed."
echo "==> Download assets / data if needed:"
echo "    (cd third_party/rmbench && bash script/_download_assets.sh && bash script/_download_data.sh)"

cat <<EOF

Environment ready.

  source ${EXAMPLE_DIR}/.venv/bin/activate
  export PYTHONPATH=${RMBENCH_ROOT}:\$PYTHONPATH
  export RMBENCH_ROOT=${RMBENCH_ROOT}

Data prep (raw → LeRobot, simulation/lerobot uv):
  bash simulation/rmbench/convert_rmbench_data_to_lerobot.sh
  # or: (cd simulation/lerobot && uv run python ../rmbench/convert_rmbench_data_to_lerobot.py cover_blocks)

Train (lbm root uv):
  DATASET=rmbench ./scripts/train.sh

Eval (two terminals; policy server is LBM):
  # T1 — LBM checkpoint server (repo-root uv)
  bash simulation/rmbench/eval_policy.sh ./checkpoints/<run>/<step>.pt
  # T2 — sim client
  bash simulation/rmbench/eval_env.sh cover_blocks

EOF
