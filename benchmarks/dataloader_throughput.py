#!/usr/bin/env python3
"""Measure custom-dump dataloader throughput on a local dataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader
from tqdm import tqdm

from lbm.action_space import ABS
from lbm.dataloader import collate_fn, dataloader_worker_init_fn
from lbm.dataloader.custom import CUSTOM_SPECS, make_custom_dataset
from lbm.dataloader.paths import datasets_root, resolve_dataset


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * q))))
    return ys[i]


def _print_profile(parent_ms: list[float], batch_size: int) -> None:
    if not parent_ms:
        print("profile: no measured batches")
        return
    print("profile  (ms / batch after warmup; mean / p50 / p95)")
    parent_total = _mean(parent_ms)
    print(
        f"  {'parent_wait':18s} {parent_total:7.2f}  "
        f"{_pct(parent_ms, 0.5):7.2f}  {_pct(parent_ms, 0.95):7.2f}   "
        f"# main-process gap between batches"
    )
    if parent_total > 0:
        print(f"  implied samp/s  {1000.0 * batch_size / parent_total:.1f} from parent_wait")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=datasets_root() / "rmbench",
        help="dataset name under datasets/ or a filesystem path",
    )
    p.add_argument("--robot-type", default="", help="spec name (default: directory name)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--warmup-batches", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument(
        "--history-length",
        type=float,
        default=0.0,
        help="image history duration in seconds (0 = current frame only)",
    )
    p.add_argument(
        "--history-freq",
        type=float,
        default=10.0,
        help="image history sampling frequency in Hz",
    )
    p.add_argument("--action-length", type=float, default=1.0, help="action window duration in seconds")
    p.add_argument(
        "--action-freq",
        type=float,
        default=None,
        help="action sampling frequency in Hz (default: dump fps)",
    )
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument("--rescan", action="store_true", help="rebuild <dataset>/.cache/episodes")
    p.add_argument("--no-mmap-frames", action="store_true")
    p.add_argument("--action-mode", default=ABS)
    p.add_argument(
        "--profile",
        action="store_true",
        help="print parent-process wait between batches",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset(str(args.dataset), required=True)
    spec_name = str(args.robot_type or dataset_root.name)
    if spec_name not in CUSTOM_SPECS:
        raise SystemExit(f"unknown spec {spec_name!r}; known: {sorted(CUSTOM_SPECS)}")

    from lbm.temporal import n_steps

    use_mmap = not args.no_mmap
    use_mmap_frames = use_mmap and not args.no_mmap_frames
    data_cfg = {
        "use_mmap": use_mmap,
        "use_mmap_frames": use_mmap_frames,
        "action_length": args.action_length,
        "history_length": args.history_length,
        "history_freq": args.history_freq,
        "action_mode": args.action_mode,
        "rescan": bool(args.rescan),
    }
    if args.action_freq is not None:
        data_cfg["action_freq"] = args.action_freq

    spec = CUSTOM_SPECS[spec_name]
    n_cams = len(spec.camera_keys)
    act_freq = float(args.action_freq or spec.fps)
    hist_freq = float(args.history_freq or spec.fps)
    n_hist = n_steps(args.history_length, hist_freq)
    n_act = n_steps(args.action_length, act_freq)
    n_jpegs = n_hist * n_cams
    print(f"dataset={dataset_root}")
    print(
        f"spec={spec_name}  batch={args.batch_size}  workers={args.num_workers}  "
        f"mmap={use_mmap}  mmap_frames={use_mmap_frames}  jpeg={spec.image_size}@q85  "
        f"history={args.history_length:g}s@{hist_freq:g}Hz→{n_hist} "
        f"action={args.action_length:g}s@{act_freq:g}Hz→{n_act} "
        f"jpegs/sample={n_jpegs}"
    )

    t0 = time.perf_counter()
    dataset = make_custom_dataset(dataset_root, spec_name, data_cfg)
    print(f"init={time.perf_counter() - t0:.1f}s  steps={len(dataset):,}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": False,
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 4 if args.num_workers > 0 else None,
    }
    if args.num_workers > 0:
        loader_kwargs["worker_init_fn"] = dataloader_worker_init_fn
    loader = DataLoader(dataset, **loader_kwargs)

    total = args.warmup_batches + args.max_batches
    n_samples = 0
    t_start = time.perf_counter()
    last = t_start
    parent_ms: list[float] = []
    with tqdm(total=total, desc="dataloader", unit="batch") as pbar:
        for batch_idx, batch in enumerate(loader):
            now = time.perf_counter()
            if batch_idx == 0:
                img = batch.get("image")
                print(
                    f"batch image={tuple(img.shape) if hasattr(img, 'shape') else type(img)} "
                    f"action={tuple(batch['action'].shape)}"
                )
            pbar.update(1)
            if batch_idx + 1 == args.warmup_batches:
                t_start = time.perf_counter()
                last = t_start
                n_samples = 0
                parent_ms.clear()
            else:
                n_samples += args.batch_size
                if batch_idx + 1 > args.warmup_batches:
                    parent_ms.append((now - last) * 1e3)
            last = now
            if batch_idx + 1 >= total:
                break

    elapsed = time.perf_counter() - t_start
    n_batches = n_samples / args.batch_size if args.batch_size else 0
    samples_s = n_samples / elapsed if elapsed > 0 else 0.0
    batches_s = n_batches / elapsed if elapsed > 0 else 0.0
    print(f"measured_batches={n_batches:.0f}  samples={n_samples}  elapsed={elapsed:.2f}s")
    print(f"throughput={samples_s:.2f} samples/s  ({batches_s:.2f} batches/s)")
    if args.profile:
        _print_profile(parent_ms, args.batch_size)


if __name__ == "__main__":
    main()
