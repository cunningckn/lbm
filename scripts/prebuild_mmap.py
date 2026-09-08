#!/usr/bin/env python3
"""Prebuild parquet trajectory mmap + JPEG frame mmap (same as rank-0 training).

Edit ``DATASETS`` below (or pass ``--dataset kai0,libero``). Missing folders are skipped.

    uv run python scripts/prebuild_mmap.py
    uv run python scripts/prebuild_mmap.py --dataset kai0
    uv run python scripts/prebuild_mmap.py --dataset kai0,libero --workers 16
    ./scripts/prebuild_mmap.sh
    DATASET=kai0 ./scripts/prebuild_mmap.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lbm.config import TrainConfig
from lbm.dataloader.mixture import load_selected_dumps
from lbm.train_cli import add_dump_list_arguments, apply_dump_list_args, dumps_from_args

DATASETS = (
    "abc",
    "agibot",
    "das_gripper",
    "droid",
    "egoverse",
    "galaxea",
    "hifi_umi",
    "hy_lance",
    "kai0",
    "libero",
    "rmbench",
    "robotwin",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prebuild parquet + JPEG mmap caches under each dump's .mmap/")
    add_dump_list_arguments(p)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="prebuild process count (0 = min(32, CPU count))",
    )
    p.add_argument("--max-episodes", type=int, default=None, help="optional cap per dump (debug)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_dump_list_args(TrainConfig(), args)
    cfg.data.use_mmap = True
    cfg.data.use_mmap_frames = True
    mix = load_selected_dumps(cfg, dumps_from_args(args, default=DATASETS), mode="train")
    n_inner = len(mix.datasets)
    print(f"prebuild mmap: {n_inner} dump(s), workers={cfg.data.mmap_prebuild_workers or 'auto'}")
    for ds in mix.datasets:
        n_ep = len(ds.records) if ds.records else len(ds.episodes)
        print(f"  {ds.spec.name}: episodes={n_ep}")
    mix.prebuild_mmap_caches(workers=cfg.data.mmap_prebuild_workers)
    print("done")


if __name__ == "__main__":
    main()
