#!/usr/bin/env bash
# Kai0 nested LeRobot v2.1
# https://huggingface.co/datasets/OpenDriveLab-org/Kai0
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" kai0 "$@"
