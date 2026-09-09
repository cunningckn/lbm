#!/usr/bin/env bash
# AgiBot World Alpha + Beta (native HDF5 + mp4)
# https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
# https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" agibot "$@"
