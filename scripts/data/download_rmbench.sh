#!/usr/bin/env bash
# RMBench official HDF5 demos (data/<task>/demo_clean/**)
# https://huggingface.co/datasets/TianxingChen/RMBench
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" rmbench "$@"
