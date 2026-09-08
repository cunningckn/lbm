#!/usr/bin/env python3
"""Build ``{dump}/.cache/episodes`` for every on-disk dump (or a named subset).

Edit ``DATASETS`` below (or pass ``--dataset kai0,libero``). Spec is the catalog name.

    uv run python scripts/build_scan_index.py
    uv run python scripts/build_scan_index.py --rescan
    uv run python scripts/build_scan_index.py --dataset kai0,libero
    ./scripts/build_scan_index.sh
    RESCAN=1 ./scripts/build_scan_index.sh
    DATASET=kai0 ./scripts/build_scan_index.sh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lbm.dataloader.catalog import on_disk_dumps
from lbm.dataloader.custom.scan import scan_root
from lbm.dataloader.custom.scan_index import cache_dir
from lbm.dataloader.custom.spec import CUSTOM_SPECS
from lbm.dataloader.paths import datasets_root
from lbm.train_cli import add_dump_list_arguments, dumps_from_args
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
    p = argparse.ArgumentParser(description="Build .cache/episodes for each dump under datasets/")
    add_dump_list_arguments(p)
    p.add_argument("--rescan", action="store_true", help="rebuild even if .cache/episodes exists")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base = Path(args.data_root).expanduser().resolve() if args.data_root else datasets_root()
    names = dumps_from_args(args, default=DATASETS)
    print(f"build scan index: root={base} dumps={len(names)} rescan={bool(args.rescan)}", flush=True)

    found = on_disk_dumps(names, base=base)
    skipped = [name for name in names if name not in {n for n, _ in found}]
    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    for name, dump in track(found, desc="scan dumps", unit="dump"):
        spec = CUSTOM_SPECS[name]
        t0 = time.perf_counter()
        try:
            records = scan_root(dump, spec, rescan=bool(args.rescan))
        except Exception as exc:
            print(f"FAIL {name}: {exc}", flush=True)
            failed.append((name, str(exc)))
            continue
        if not records:
            print(f"FAIL {name}: scan returned 0 episodes", flush=True)
            failed.append((name, "scan returned 0 episodes"))
            continue
        steps = sum(max(int(r.n_frames), 0) for r in records)
        print(
            f"ok {name}: episodes={len(records):,} steps={steps:,} "
            f"{time.perf_counter() - t0:.1f}s -> {cache_dir(dump)}",
            flush=True,
        )
        ok.append(name)

    print(f"done ok={len(ok)} skip={len(skipped)} fail={len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)
    if not ok:
        raise SystemExit("no dumps indexed")


if __name__ == "__main__":
    main()
