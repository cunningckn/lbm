#!/usr/bin/env bash
# ABC-130k (already MCAP)
# https://huggingface.co/datasets/XDOF/ABC-130k
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" abc "$@"
