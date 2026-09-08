#!/usr/bin/env python3
"""Megatron-FSDP training on synthetic batches via ``train_loop.main``.

    torchrun --standalone --nproc_per_node=8 examples/train_fsdp.py
    torchrun --standalone --nproc_per_node=2 examples/train_fsdp.py --steps 4 --batch-size 2
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
    args = parse_args(default_dataset="", default_fake=True)
    args.fsdp = True
    train_main(build_train_config(args))


if __name__ == "__main__":
    main()
