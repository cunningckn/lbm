#!/usr/bin/env bash
# Convert every mix dump from datasets/raw/<name> into datasets/<name>.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DIR}/process.sh" --all "$@"
