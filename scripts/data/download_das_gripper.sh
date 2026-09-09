#!/usr/bin/env bash
# DAS Gripper — public sample HDF5 (full 10Kh corpus is MCAP)
# https://huggingface.co/datasets/genrobot2025/DAS-Sample-Data
# https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" das_gripper "$@"
