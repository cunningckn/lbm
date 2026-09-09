#!/usr/bin/env bash
# RoboTwin unified LeRobot v3 (what LBM reads)
# https://huggingface.co/datasets/lerobot/robotwin_unified
# Official HDF5 (optional): https://huggingface.co/datasets/TianxingChen/RoboTwin2.0
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" robotwin "$@"
