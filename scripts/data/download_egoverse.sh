#!/usr/bin/env bash
# EgoVerse Aria zarr via official S3 sync (not Hugging Face)
# https://github.com/GaTech-RL2/EgoVerse
# python egomimic/scripts/data_download/sync_s3.py --local-dir DEST --filters aria-all
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" egoverse "$@"
