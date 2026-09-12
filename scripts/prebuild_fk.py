#!/usr/bin/env python3
"""Precompute joint→EEF proprio into ``{dump}/.cache/fk``.

Writes per-episode ``state.npy`` / ``action.npy`` after FK. Training loads them
when ``--action-kind eef``. Dumps with no registered chain are skipped.

    uv run python scripts/prebuild_fk.py
    uv run python scripts/prebuild_fk.py --dataset kai0
    uv run python scripts/prebuild_fk.py --dataset kai0,agibot --workers 16
    uv run python scripts/prebuild_fk.py --force
    ./scripts/prebuild_fk.sh
    DATASET=kai0 ./scripts/prebuild_fk.sh
    FORCE=1 WORKERS=16 DATASET=agibot ./scripts/prebuild_fk.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lbm.config import TrainConfig
from lbm.dataloader.custom.datasets import dataset_names, dataset_spec
from lbm.dataloader.custom.fk_cache import needs_joint_fk
from lbm.dataloader.mixture import load_selected_dumps
from lbm.train_cli import add_dump_list_arguments, apply_dump_list_args, dumps_from_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute joint→EEF into each dump's .cache/fk/")
    add_dump_list_arguments(p)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="process count (0 = min(32, CPU count))",
    )
    p.add_argument("--force", action="store_true", help="rebuild even if an episode cache exists")
    p.add_argument("--max-episodes", type=int, default=None, help="optional cap per dump (debug)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_dump_list_args(TrainConfig(), args)
    cfg.data.action_kind = "eef"
    cfg.data.use_mmap = True
    cfg.data.use_mmap_frames = False
    cfg.data.mmap_prebuild = False
    defaults = tuple(name for name in dataset_names() if needs_joint_fk(dataset_spec(name)))
    mix = load_selected_dumps(cfg, dumps_from_args(args, default=defaults), mode="train")
    n_inner = len(mix.datasets)
    workers = cfg.data.mmap_prebuild_workers
    print(f"prebuild fk: {n_inner} dump(s), workers={workers or 'auto'} force={bool(args.force)}")
    for ds in mix.datasets:
        n_ep = len(ds.records) if ds.records else len(ds.episodes)
        print(f"  {ds.spec.name}: episodes={n_ep}")
    mix.prebuild_fk_caches(workers=workers, force=bool(args.force))
    print("done")


if __name__ == "__main__":
    main()
