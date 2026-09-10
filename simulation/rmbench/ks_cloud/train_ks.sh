#!/bin/bash
# Cloud entrypoint for RMBench MEM finetune (openpi / JAX).
# Submit via: python examples/rmbench/ks_cloud/submit_job.py -g 4
#
# NOTE: submit_job.py copies this script into logs/.../ before running.
# Do NOT derive OPENPI_ROOT from BASH_SOURCE; submit always `cd`s to openpi root first.
#
# Source bashrc BEFORE set -u: non-interactive shells leave PS1 unset, and sourcing
# bashrc under `set -u` aborts immediately (empty log, ~5s failed).
source ~/.bashrc 2>/dev/null || true
set -euo pipefail

OPENPI_ROOT="$(pwd)"
KS_CLOUD_DIR="${OPENPI_ROOT}/examples/rmbench/ks_cloud"
cd "${OPENPI_ROOT}"

# Proxy for uv / network downloads on KS.
export http_proxy="http://10.0.0.222:8888"
export https_proxy="http://10.0.0.222:8888"

# Venv under /root (node-local). CPython: uv default install dir.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${HOME}/.local/share/uv/python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/root/uv_envs/openpi}"
export PATH="${HOME}/.local/bin:${PATH}"

VENV_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [ ! -x "${VENV_PYTHON}" ]; then
  echo "ERROR: missing venv python at ${VENV_PYTHON}"
  echo "  export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
  echo "  uv python install 3.11"
  echo "  uv venv --clear --python 3.11 \"\${UV_PROJECT_ENVIRONMENT}\""
  echo "  GIT_LFS_SKIP_SMUDGE=1 uv sync"
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run)
else
  RUN=("${VENV_PYTHON}")
fi

echo "PYTHON=${VENV_PYTHON}"
echo "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "OPENPI_ROOT=${OPENPI_ROOT}"

"${VENV_PYTHON}" "${KS_CLOUD_DIR}/wandb_login.py"

export HF_ENDPOINT=https://hf-mirror.com
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export OPENPI_USE_LOCAL_CHECKPOINTS="${OPENPI_USE_LOCAL_CHECKPOINTS:-1}"

# Optional: copy dataset off KPFS to avoid random-read bottleneck.
# COPY_DATA_TO_LOCAL=0  -> use KPFS path directly
# COPY_DATA_TO_LOCAL=1  -> rsync to /root/local_data/rmbench (node-local disk)
# COPY_DATA_TO_LOCAL=2  -> rsync to /dev/shm/rmbench (tmpfs; read from RAM)
COPY_DATA_TO_LOCAL=2
DATA_SRC="${OPENPI_ROOT}/data/rmbench"
DATA_LOCAL_ROOT="${DATA_LOCAL_ROOT:-/root/local_data}"
DATA_SHM_ROOT="${DATA_SHM_ROOT:-/dev/shm}"

if [ ! -d "${DATA_SRC}" ] && [ "${COPY_DATA_TO_LOCAL}" != "0" ]; then
  echo "ERROR: dataset not found at ${DATA_SRC}"
  exit 1
fi

case "${COPY_DATA_TO_LOCAL}" in
  1)
    DATA_DST="${DATA_LOCAL_ROOT}/rmbench"
    mkdir -p "${DATA_LOCAL_ROOT}"
    echo "Copying dataset to local disk: ${DATA_SRC} -> ${DATA_DST}"
    # -a: archive; --info=progress2: overall progress (GNU rsync)
    rsync -a --info=progress2 "${DATA_SRC}/" "${DATA_DST}/"
    DATA_REPO_ID="${DATA_DST}"
    ;;
  2)
    DATA_DST="${DATA_SHM_ROOT}/rmbench"
    if [ ! -d "${DATA_SHM_ROOT}" ]; then
      echo "ERROR: tmpfs root not found at ${DATA_SHM_ROOT}"
      exit 1
    fi
    mkdir -p "${DATA_DST}"
    echo "Copying dataset to memory (tmpfs): ${DATA_SRC} -> ${DATA_DST}"
    # -a: archive; --info=progress2: overall progress (GNU rsync)
    rsync -a --info=progress2 "${DATA_SRC}/" "${DATA_DST}/"
    DATA_REPO_ID="${DATA_DST}"
    ;;
  *)
    echo "Skipping local copy; using ${DATA_SRC}"
    DATA_REPO_ID="${DATA_SRC}"
    ;;
esac
echo "DATA_REPO_ID=${DATA_REPO_ID}"

# Keep exp_name parseable by submit_job.py (also accepts --exp-name=...).
exp_name="20260720_rmbench_mem_ki_language_bs32_hl18_hs50_30k"

"${RUN[@]}" scripts/train.py pi05_rmbench_mem_ki_language \
  --exp-name="${exp_name}" \
  --weight-loader.params-path=./checkpoints/pi05_base/params \
  --data.repo-id="${DATA_REPO_ID}" \
  --data.assets.asset-id=data/rmbench \
  --batch-size=32 \
  --fsdp-devices=8 \
  --num-workers=32 \
  --model.image-history-length=18 \
  --model.image-history-stride=50 \
  --num-train-steps=30000 \
  --lr-schedule.warmup-steps=1000 \
  --lr-schedule.decay-steps=30000 \
  --log-interval=100 \
  --val-interval=1000 \
  --val-episode-ratio=0.1 \
  --val-batch-size=32 \
  --save-interval=10000 \
  --overwrite
