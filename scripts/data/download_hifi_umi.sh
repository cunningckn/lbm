#!/usr/bin/env bash
# HiFi-UMI-2K (already LeRobot v3)
# https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" hifi_umi "$@"
