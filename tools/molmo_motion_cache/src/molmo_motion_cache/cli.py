"""Command-line entry points for building, checking, and timing DROID caches."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common import byte_size, read_json, utc_now, verify_checksum_manifest, write_json
from .droid import build_droid_cache
from .reader import MMapDroidReader, RawTarDroidReader


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _verify_values(
    source_root: Path, reader: MMapDroidReader, checks: int
) -> dict[str, Any]:
    sample_ids = reader.sample_ids
    count = min(max(checks, 0), len(sample_ids))
    if not count:
        return {"checked_source_records": 0, "checked_sample_ids": []}
    positions = np.linspace(0, len(sample_ids) - 1, num=count, dtype=np.int64)
    raw_reader = RawTarDroidReader(source_root, reader)
    checked_ids: list[str] = []
    try:
        for position in positions.tolist():
            sample_id = sample_ids[position]
            raw = raw_reader.get_full(sample_id)
            cached = reader.get_full(sample_id)
            for name in (
                "points2d",
                "points3d",
                "visibility2d",
                "visibility3d",
                "intrinsics_measured",
                "intrinsics_ds",
                "extrinsics",
            ):
                if not _array_equal(raw[name], cached[name]):
                    raise ValueError(f"source/cache mismatch for {sample_id}: {name}")
            checked_ids.append(sample_id)
    finally:
        raw_reader.close()
    return {"checked_source_records": len(checked_ids), "checked_sample_ids": checked_ids}


def verify_cache(
    source_root: str | Path,
    output: str | Path,
    *,
    checks: int,
    verify_hashes: bool,
) -> dict[str, Any]:
    output_path = Path(output).resolve()
    readiness = output_path / "PILOT_READY.json"
    if not readiness.is_file():
        raise FileNotFoundError(
            f"cache has no PILOT_READY.json and is not a completed pilot: {output_path}"
        )
    reader = MMapDroidReader(output_path)
    result: dict[str, Any] = {
        "status": "passed",
        "output": str(output_path),
        "records_indexed": len(reader.sample_ids),
        "verified_at": utc_now(),
        **_verify_values(Path(source_root).resolve(), reader, checks),
    }
    if verify_hashes:
        result["checksum_validation"] = verify_checksum_manifest(output_path)
    return result


def _consume_window(window: dict[str, np.ndarray]) -> float:
    """Force numerical data access so timing is not only index lookup."""

    checksum = 0.0
    for name in (
        "points2d",
        "points3d",
        "visibility2d",
        "visibility3d",
        "intrinsics_measured",
        "intrinsics_ds",
        "extrinsics",
    ):
        array = window[name]
        if np.issubdtype(array.dtype, np.floating):
            checksum += float(np.nan_to_num(array, nan=0.0).sum(dtype=np.float64))
        else:
            checksum += float(array.sum(dtype=np.int64))
    return checksum


def _measure(
    get_window: Callable[..., dict[str, np.ndarray]],
    operations: list[tuple[str, int]],
    frames: int,
    points: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    checksum = 0.0
    started = time.perf_counter()
    for sample_id, start in operations:
        tick = time.perf_counter()
        checksum += _consume_window(
            get_window(sample_id, start=start, frames=frames, points=points)
        )
        latencies_ms.append((time.perf_counter() - tick) * 1000.0)
    elapsed = time.perf_counter() - started
    count = len(operations)
    return {
        "samples": count,
        "elapsed_seconds": elapsed,
        "samples_per_second": count / elapsed if elapsed else math.inf,
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "checksum": checksum,
    }


def benchmark_cache(
    source_root: str | Path,
    output: str | Path,
    *,
    samples: int,
    warmup: int,
    frames: int,
    points: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0 or warmup < 0:
        raise ValueError("samples must be positive and warmup cannot be negative")
    reader = MMapDroidReader(output)
    sample_ids = reader.sample_ids
    if not sample_ids:
        raise RuntimeError("cache has no records")
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(sample_ids), size=samples + warmup)
    operations: list[tuple[str, int]] = []
    for index in chosen.tolist():
        sample_id = sample_ids[index]
        max_start = max(0, int(reader.tracks[sample_id]["num_frames"]) - frames)
        start = int(rng.integers(0, max_start + 1))
        operations.append((sample_id, start))
    warm_operations = operations[:warmup]
    timed_operations = operations[warmup:]
    raw_reader = RawTarDroidReader(source_root, reader)
    try:
        for sample_id, start in warm_operations:
            _consume_window(raw_reader.get_window(sample_id, start=start, frames=frames, points=points))
        for sample_id, start in warm_operations:
            _consume_window(reader.get_window(sample_id, start=start, frames=frames, points=points))
        raw_metrics = _measure(raw_reader.get_window, timed_operations, frames, points)
        mmap_metrics = _measure(reader.get_window, timed_operations, frames, points)
    finally:
        raw_reader.close()
    if not np.isclose(raw_metrics["checksum"], mmap_metrics["checksum"], rtol=1e-12, atol=1e-6):
        raise ValueError("benchmark source/cache checksums differ")
    raw_rate = raw_metrics["samples_per_second"]
    mmap_rate = mmap_metrics["samples_per_second"]
    return {
        "status": "completed",
        "measured_at": utc_now(),
        "cache_output": str(Path(output).resolve()),
        "method": (
            "warm-cache randomized window reads; source path decodes paired NPZ "
            "members from tar and caches parsed camera JSON; cache path reads NPY "
            "memmaps after one Parquet index load"
        ),
        "parameters": {
            "samples": samples,
            "warmup": warmup,
            "frames": frames,
            "points": points,
            "seed": seed,
        },
        "raw_tar_npz_json": raw_metrics,
        "mmap_npy_parquet_index": mmap_metrics,
        "throughput_speedup": mmap_rate / raw_rate if raw_rate else math.inf,
        "checksums_equal": True,
        "limitations": [
            "OS page cache was warm; this is not a cold-disk benchmark.",
            "The DROID pilot has no reconstructed video frames, so MP4 frame decode is not measured.",
            "Both paths perform the same NumPy window materialization and checksum."
        ],
    }


def inspect_droid_source(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root).resolve()
    droid = root / "droid"
    completed = sorted(
        list((droid / "tracks").glob("tracks-*.tar"))
        + list((droid / "camera").glob("camera-*.tar"))
    )
    annotations = sorted((droid / "annotations").glob("*.json"))
    partials = sorted(
        list((droid / "tracks").glob(".*.part"))
        + list((droid / "camera").glob(".*.part"))
    )
    split = read_json(droid / "annotations" / "droid_split.json")
    return {
        "source_root": str(root),
        "completed_droid_files": len(completed) + len(annotations),
        "completed_droid_bytes": byte_size([*completed, *annotations]),
        "completed_track_camera_bytes": byte_size(completed),
        "incomplete_transfer_files_observed": len(partials),
        "incomplete_transfer_bytes_observed": byte_size(partials),
        "official_split_counts": {
            key: len(value)
            for key, value in split.items()
            if isinstance(value, list)
        },
    }


def _add_common_source_output(command: argparse.ArgumentParser) -> None:
    command.add_argument("--source-root", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molmo-motion-cache")
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build-droid", help="build a DROID mmap cache")
    _add_common_source_output(build)
    build.add_argument(
        "--limit",
        type=int,
        default=None,
        help="positive pilot record count; omit to process all currently complete DROID records",
    )
    build.add_argument("--shard-size", type=int, default=512)
    build.add_argument("--checks", type=int, default=16)

    verify = subcommands.add_parser("verify", help="check a completed DROID pilot")
    _add_common_source_output(verify)
    verify.add_argument("--checks", type=int, default=32)
    verify.add_argument("--verify-hashes", action="store_true")
    verify.add_argument("--report", type=Path, default=None)

    benchmark = subcommands.add_parser(
        "benchmark", help="compare raw tar/NPZ reads to mmap-cache reads"
    )
    _add_common_source_output(benchmark)
    benchmark.add_argument("--samples", type=int, default=512)
    benchmark.add_argument("--warmup", type=int, default=32)
    benchmark.add_argument("--frames", type=int, default=8)
    benchmark.add_argument("--points", type=int, default=32)
    benchmark.add_argument("--seed", type=int, default=20260914)
    benchmark.add_argument("--report", type=Path, default=None)

    inspect = subcommands.add_parser(
        "inspect-droid", help="summarize only completed DROID source files"
    )
    inspect.add_argument("--source-root", required=True, type=Path)
    inspect.add_argument("--report", type=Path, default=None)
    return parser


def _write_optional_report(result: dict[str, Any], report: Path | None) -> None:
    if report is not None:
        write_json(report.resolve(), result)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "build-droid":
        result = build_droid_cache(
            arguments.source_root,
            arguments.output,
            limit=arguments.limit,
            shard_size=arguments.shard_size,
            checks=arguments.checks,
        )
    elif arguments.command == "verify":
        result = verify_cache(
            arguments.source_root,
            arguments.output,
            checks=arguments.checks,
            verify_hashes=arguments.verify_hashes,
        )
        _write_optional_report(result, arguments.report)
    elif arguments.command == "benchmark":
        result = benchmark_cache(
            arguments.source_root,
            arguments.output,
            samples=arguments.samples,
            warmup=arguments.warmup,
            frames=arguments.frames,
            points=arguments.points,
            seed=arguments.seed,
        )
        _write_optional_report(result, arguments.report)
    elif arguments.command == "inspect-droid":
        result = inspect_droid_source(arguments.source_root)
        _write_optional_report(result, arguments.report)
    else:
        raise AssertionError(f"unhandled command: {arguments.command}")
    _json_print(result)
    return 0
