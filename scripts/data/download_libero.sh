#!/usr/bin/env bash
# LIBERO (already the HF LeRobot dump LBM reads)
# https://huggingface.co/datasets/physical-intelligence/libero
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" libero "$@"
