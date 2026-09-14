#!/usr/bin/env bash
# Compatibility entrypoint used by the job submitter. Runtime/path handling is
# centralized in convert_50cpu.sh so local and submitted jobs behave the same.
exp_name="lbm_convert_${DATASET:-dataset}"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${ROOT}/scripts/data/convert_50cpu.sh"
