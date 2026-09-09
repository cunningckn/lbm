#!/usr/bin/env bash
# Hy-Embodied-0.5-VLA-Data (already Lance)
# https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" hy_lance "$@"
