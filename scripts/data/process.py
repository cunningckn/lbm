#!/usr/bin/env python3
"""Turn an official download into the dump layout LBM already reads.

This is not scan / FK / mmap / norm. Those stay in scripts/prebuild_*.py
and scripts/compute_norm.py.

    ./scripts/data/process.sh galaxea
    uv run python scripts/data/process.py --dataset rmbench
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from catalog import DUMPS, NAMES, dump  # noqa: E402
from converters import agibot as agibot_c  # noqa: E402
from converters import das as das_c  # noqa: E402
from converters import detect, layout  # noqa: E402
from converters import galaxea as galaxea_c  # noqa: E402
from converters import rmbench as rmbench_c  # noqa: E402


def datasets_dir() -> Path:
    return Path(__import__("os").environ.get("LBM_DATASETS") or (ROOT / "datasets")).expanduser()


def raw_dir() -> Path:
    import os

    return Path(os.environ.get("LBM_DATA_RAW") or (datasets_dir() / "raw")).expanduser()


def dest_path(name: str) -> Path:
    return datasets_dir() / name


def src_path(name: str) -> Path:
    raw = raw_dir() / name
    dest = dest_path(name)
    if raw.exists():
        return raw
    return dest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official download → LBM dump layout")
    p.add_argument("dataset", nargs="?", help="dump name (or comma-separated)")
    p.add_argument("--dataset", dest="dataset_flag", default=None, help="same as positional")
    p.add_argument("--all", action="store_true", help="every mix dump")
    p.add_argument("--raw", type=Path, default=None, help="override official download dir")
    p.add_argument("--dest", type=Path, default=None, help="override LBM dump dir")
    p.add_argument("--force", action="store_true", help="re-run conversion even if dest looks ready")
    return p.parse_args(argv)


def names_from_args(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(NAMES)
    text = args.dataset_flag or args.dataset or ""
    names = [n.strip() for n in text.split(",") if n.strip()]
    if not names:
        raise SystemExit("usage: process.py [--all] <dump-name>")
    unknown = [n for n in names if n not in DUMPS]
    if unknown:
        raise SystemExit(f"unknown dump {unknown}; expected {', '.join(NAMES)}")
    return names


def process_one(name: str, *, src: Path, dest: Path, force: bool = False) -> Path:
    spec = dump(name)
    print(f"[{name}] url: {spec['urls'][0]}")
    print(f"[{name}] process: {spec['process']} — {spec.get('process_note', '')}")
    print(f"[{name}] src={src} dest={dest}")
    if dest.exists() and not force:
        kind = detect.ready_kind(dest, name)
        if kind:
            print(f"[{name}] already LBM dump ({kind}), skip")
            return dest
    src = src if src.exists() else dest
    if not src.exists():
        local = spec.get("local") or ""
        if local and Path(local).exists():
            print(f"[{name}] no raw download; local processed dump at {local}")
            layout.ensure_dest(Path(local), dest, force=force)
            return dest
        raise SystemExit(f"[{name}] missing official download at {raw_dir() / name} and dest {dest}")
    kind = spec["process"]
    if kind == "none":
        if dest.exists() and dest.resolve() == src.resolve():
            print(f"[{name}] download is already the dump")
            return dest
        layout.ensure_dest(src, dest, force=force)
        return dest
    if kind == "galaxea_extract":
        return galaxea_c.extract_galaxea(src, dest, force=force)
    if kind == "agibot_layout":
        return agibot_c.layout_agibot(src, dest, force=force)
    if kind == "rmbench_hdf5":
        return rmbench_c.convert_rmbench(src, dest, force=force)
    if kind == "das_slim":
        return das_c.convert_das(src, dest, force=force)
    raise SystemExit(f"[{name}] unknown process {kind!r}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    names = names_from_args(args)
    for name in names:
        src = Path(args.raw).expanduser() if args.raw else src_path(name)
        dest = Path(args.dest).expanduser() if args.dest else dest_path(name)
        if len(names) > 1 and args.raw:
            src = Path(args.raw).expanduser() / name
        if len(names) > 1 and args.dest:
            dest = Path(args.dest).expanduser() / name
        process_one(name, src=src, dest=dest, force=bool(args.force))
    print("done")


if __name__ == "__main__":
    main()
