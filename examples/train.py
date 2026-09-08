#!/usr/bin/env python3
"""Minimal training loop on synthetic batches (no real images / CLIP).

    PYTHONPATH=src python examples/train.py
    PYTHONPATH=src python examples/train.py --steps 8 --batch-size 1

Real rmbench data lives in ``scripts/train.py`` / ``scripts/train.sh``.
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
    train_main(build_train_config(parse_args(default_dataset="", default_fake=True)))


if __name__ == "__main__":
    main()
