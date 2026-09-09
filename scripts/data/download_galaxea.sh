#!/usr/bin/env bash
# Galaxea Open-World — download lerobot/*.tar.gz only
# https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh" galaxea "$@"
