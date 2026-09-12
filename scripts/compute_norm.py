#!/usr/bin/env python3
"""Compute state/action mean/std/min/max/q01/q99 for normalize/unnormalize.

Writes ``norm_stats_{kind}_{rep}_{format}_{freq}hz.json`` (abs / delta / file-delta)
or ``norm_stats_{kind}_{rep}_{format}_{freq}hz_{length}s.json`` (rel) next to each dump.
Training loads the file that matches that dump's freq / length / mode.
LeRobot dumps read proprio from parquet mmap (same as training). Pass ``--mmap``
to also prebuild JPEG frame caches in the same run.

All registered datasets are selected by default (or pass ``--dataset kai0,libero``). Missing folders are skipped.

    uv run python scripts/compute_norm.py
    uv run python scripts/compute_norm.py --dataset kai0
    uv run python scripts/compute_norm.py --dataset kai0 --mmap --workers 16
    ./scripts/compute_norm.sh
    DATASET=kai0 ./scripts/compute_norm.sh
    MMAP=1 DATASET=kai0 ./scripts/compute_norm.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lbm.action_space import resolve_action_space
from lbm.config import TrainConfig, add_temporal_arguments, apply_temporal_args
from lbm.dataloader.mixture import load_selected_dumps
from lbm.train_cli import add_dump_list_arguments, apply_dump_list_args, dumps_from_args
from lbm.utils.preprocess import compute_norm_stats, dump_norm_stats_path, save_norm_stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-dump state/action q01/q99 norm stats")
    add_dump_list_arguments(p)
    add_temporal_arguments(p)
    p.add_argument("--action-mode", default="delta", help="delta (default), rel, or abs")
    p.add_argument("--action-kind", default=None, help="eef: joint→EEF via FK (aloha, franka)")
    p.add_argument(
        "--action-format",
        default=None,
        help="eef packed pose: default, xyz+rotvec, xyz+rot6d, xyz+quat",
    )
    p.add_argument(
        "--output",
        default="",
        help="write this file (single dump only). default: freq/length name next to the dump",
    )
    p.add_argument(
        "--mmap",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="also prebuild parquet + JPEG mmap (same as prebuild_mmap.py)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="mmap prebuild process count (0 = min(32, CPU count))",
    )
    p.add_argument("--max-episodes", type=int, default=None, help="optional cap per dump (debug)")
    return p.parse_args(argv)


def _output_path(dataset, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    root = getattr(dataset, "root", None)
    if root is not None:
        slices = resolve_action_space(
            dataset.spec,
            dataset.action_mode,
            action_kind=getattr(dataset, "action_kind", None),
            action_format=getattr(dataset, "action_format", None),
        )
        return dump_norm_stats_path(root, dataset.action_freq, dataset.action_length, slices)
    raise SystemExit(f"cannot infer dump dir for {dataset.spec.name}; pass --output")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = apply_dump_list_args(TrainConfig(), args)
    apply_temporal_args(cfg.model, args)
    cfg.data.action_mode = args.action_mode
    cfg.data.action_kind = args.action_kind
    cfg.data.action_format = str(getattr(args, "action_format", None) or "")
    if getattr(args, "action_freq", None) is not None:
        cfg.data.override_action_freq = True
    cfg.data.use_mmap = True
    cfg.data.use_mmap_frames = True
    mix = load_selected_dumps(cfg, dumps_from_args(args), mode="train")
    if args.mmap:
        n_inner = len(mix.datasets)
        print(f"prebuild mmap: {n_inner} dump(s), workers={cfg.data.mmap_prebuild_workers or 'auto'}")
        mix.prebuild_mmap_caches(workers=cfg.data.mmap_prebuild_workers)
    singles = list(mix.datasets)
    if args.output and len(singles) > 1:
        raise SystemExit("--output is for one dump; omit it to write each dump's norm_stats.json")
    many = len(singles) > 1
    dumps = singles
    if many:
        from lbm.utils.progress import track

        dumps = track(singles, desc="norm dumps", unit="dump")
    for dataset in dumps:
        payload = compute_norm_stats(dataset, leave=not many)
        path = _output_path(dataset, args.output)
        save_norm_stats(path, payload)
        n = payload["norm_stats"]["actions"]["count"]
        print(f"{payload.get('spec') or '?'}  frames={n}  -> {path}")


if __name__ == "__main__":
    main()
