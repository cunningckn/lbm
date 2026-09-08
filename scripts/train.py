#!/usr/bin/env python3
"""Train LBM on a LeRobot dataset or mixture.

    uv run python scripts/train.py --dataset kai0
    uv run python scripts/train.py --dataset rmbench --robot-type rmbench
    DATASET=kai0 ./scripts/train.sh
    torchrun --standalone --nproc_per_node=8 scripts/train.py --fsdp --data-mix all
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lbm.train_cli import build_train_config, parse_args
from lbm.train_loop import main as train_main


def main() -> None:
    args = parse_args()
    if not args.fake_data and not args.dataset and not args.data_mix:
        raise SystemExit(
            "pass --dataset NAME (under datasets/) or a filesystem path, "
            "or --data-mix NAME, or --fake-data"
        )
    train_main(build_train_config(args))


if __name__ == "__main__":
    main()
