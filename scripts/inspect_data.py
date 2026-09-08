#!/usr/bin/env python3
"""Read dumps through the training DataLoader and dump the first batch.

Vision → PNG + MP4, state/action → plots, language → txt. Then keeps
iterating (no train) so IO can be timed.

Edit ``DATASETS`` below (or pass ``--dataset rmbench``). Missing folders are skipped.

    uv run python scripts/inspect_data.py
    uv run python scripts/inspect_data.py --dataset rmbench
    uv run python scripts/inspect_data.py --dataset rmbench --max-batches 1
    ./scripts/inspect_data.sh
    DATASET=rmbench ./scripts/inspect_data.sh
    MAX_BATCHES=8 HISTORY_LENGTH=1 DATASET=rmbench ./scripts/inspect_data.sh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lbm.config import TrainConfig, add_temporal_arguments, apply_temporal_args
from lbm.dataloader.mixture import load_selected_dumps
from lbm.dataloader.pad import collate_fn, dataloader_worker_init_fn
from lbm.train_cli import add_dump_list_arguments, apply_dump_list_args, dumps_from_args
from lbm.utils.batch_dump import describe_loader_batch, save_loader_batch, video_fps_from_config
from lbm.utils.progress import track

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
    p = argparse.ArgumentParser(description="Loop the training DataLoader and save the first batch")
    add_dump_list_arguments(p)
    add_temporal_arguments(p)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="stop after N batches (0 = iterate the whole loader)",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "inspect_data"),
        help="directory for the first-batch dump",
    )
    p.add_argument("--fps", type=float, default=0.0, help="video fps (0 = history_freq / dump fps)")
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument("--rescan", action="store_true", help="rebuild <dataset>/.cache/episodes")
    p.add_argument("--max-episodes", type=int, default=None, help="optional cap per dump (debug)")
    p.add_argument("--shuffle", action="store_true", help="shuffle like training (default: off)")
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers)
    apply_dump_list_args(cfg, args)
    apply_temporal_args(cfg.model, args)
    if getattr(args, "action_freq", None) is not None:
        cfg.data.override_action_freq = True
    cfg.data.use_mmap = not args.no_mmap
    cfg.data.use_mmap_frames = not args.no_mmap
    cfg.data.mmap_prebuild = False
    cfg.data.rescan = bool(args.rescan)
    return cfg


def _make_loader(dataset, cfg: TrainConfig, *, shuffle: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": False,
        "drop_last": False,
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = cfg.data.prefetch_factor
        kwargs["worker_init_fn"] = dataloader_worker_init_fn
    return DataLoader(dataset, **kwargs)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = _build_config(args)
    dataset = load_selected_dumps(cfg, dumps_from_args(args, default=DATASETS), mode="train")
    n_inner = len(getattr(dataset, "datasets", [dataset]))
    print(
        f"dataset samples={len(dataset)} dumps={n_inner} "
        f"batch_size={cfg.batch_size} workers={cfg.num_workers} "
        f"action={cfg.model.action_length:g}s history={cfg.model.history_length:g}s"
    )
    loader = _make_loader(dataset, cfg, shuffle=bool(args.shuffle))
    fps = float(args.fps) if args.fps and args.fps > 0 else video_fps_from_config(cfg, dataset)
    out_dir = Path(args.output_dir).expanduser()
    first_dir = out_dir / "first_batch"

    n_limit = int(args.max_batches) if args.max_batches and args.max_batches > 0 else None
    n_done = 0
    n_samples = 0
    t0 = time.perf_counter()
    first_saved = False
    iterable = track(loader, desc="read", unit="batch", total=n_limit)
    try:
        for batch in iterable:
            if not first_saved:
                print(describe_loader_batch(batch))
                save_loader_batch(batch, first_dir, fps=fps)
                first_saved = True
                print(f"saved first batch -> {first_dir}")
            n_done += 1
            action = batch["action"]
            n_samples += int(action.shape[0] if torch.is_tensor(action) else len(action))
            if n_limit is not None and n_done >= n_limit:
                break
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    elapsed = max(time.perf_counter() - t0, 1e-9)
    print(
        f"done batches={n_done} samples={n_samples} "
        f"{n_done / elapsed:.2f} batch/s  {n_samples / elapsed:.2f} sample/s  {elapsed:.1f}s"
    )
    if not first_saved:
        raise SystemExit("loader yielded no batches")


if __name__ == "__main__":
    main()
