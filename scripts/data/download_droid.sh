#!/usr/bin/env bash
# DROID 1.0.1 (already LeRobot v2.1)
# https://huggingface.co/datasets/lerobot/droid_1.0.1
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" droid "$@"
