"""Comparable raw-NPZ versus mmap benchmarks for generic subsets."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .adapters import (
    GENERIC_SUBSETS,
    Candidate,
    NormalizedSample,
    candidate_seeds,
    load_sample as _load_sample,
    resolve_candidates,
)
from .archives import close_cached_archives
from .common import utc_now
from .generic_reader import MMapMotionReader


Operation = tuple[Candidate, str, int]


def _window(
    sample: NormalizedSample,
    object_id: str,
    *,
    start: int,
    frames: int,
    points: int,
) -> dict[str, np.ndarray]:
    obj = next(value for value in sample.objects if value.object_id == object_id)
    total_frames, total_points = obj.points2d.shape[:2]
    selected_frames = min(frames, total_frames)
    start = min(max(start, 0), total_frames - selected_frames)
    stop = start + selected_frames
    point_indices = np.linspace(
        0, total_points - 1, num=min(points, total_points), dtype=np.int64
    )
    camera = sample.camera
    result = {
        "points2d": np.ascontiguousarray(obj.points2d[start:stop, point_indices]),
        "points3d": np.ascontiguousarray(obj.points3d[start:stop, point_indices]),
        "visibility2d": np.ascontiguousarray(obj.visibility2d[start:stop, point_indices]),
        "visibility3d": np.ascontiguousarray(obj.visibility3d[start:stop, point_indices]),
        "camera_poses": np.ascontiguousarray(camera.poses),
        "camera_pose_indices": np.ascontiguousarray(camera.pose_indices),
        "camera_intrinsics_dynamic": np.ascontiguousarray(camera.dynamic_intrinsics),
        "camera_intrinsic_indices": np.ascontiguousarray(camera.intrinsic_indices),
        "camera_intrinsics_static": np.ascontiguousarray(camera.static_intrinsics),
        "clip_frame_indices": np.arange(start, stop, dtype=np.int64),
        "source_point_indices": np.ascontiguousarray(obj.source_point_indices[point_indices]),
    }
    if obj.trust_weights is not None:
        result["trust_weights"] = np.ascontiguousarray(
            obj.trust_weights[start:stop, point_indices]
        )
    if obj.keep_mask is not None:
        result["keep_mask"] = np.ascontiguousarray(obj.keep_mask)
    return result


def _consume(value: dict[str, np.ndarray]) -> float:
    checksum = 0.0
    for array in value.values():
        if np.issubdtype(array.dtype, np.floating):
            checksum += float(np.nan_to_num(array, nan=0.0).sum(dtype=np.float64))
        else:
            checksum += float(array.sum(dtype=np.int64))
    return checksum


def _measure(
    getter: Callable[[Operation], dict[str, np.ndarray]], operations: list[Operation]
) -> dict[str, Any]:
    latencies: list[float] = []
    checksum = 0.0
    started = time.perf_counter()
    for operation in operations:
        tick = time.perf_counter()
        checksum += _consume(getter(operation))
        latencies.append((time.perf_counter() - tick) * 1000.0)
    elapsed = time.perf_counter() - started
    return {
        "samples": len(operations),
        "elapsed_seconds": elapsed,
        "samples_per_second": len(operations) / elapsed if elapsed else math.inf,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "latency_ms_mean": float(np.mean(latencies)),
        "checksum": checksum,
    }


def benchmark_generic_cache(
    source_root: str | Path,
    output: str | Path,
    dataset: str,
    *,
    source_records_per_track_kind: int,
    samples: int,
    warmup: int,
    frames: int,
    points: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    if dataset not in GENERIC_SUBSETS:
        raise ValueError(f"unsupported generic subset: {dataset}")
    if source_records_per_track_kind <= 0 or samples <= 0 or warmup < 0:
        raise ValueError("source records/samples must be positive and warmup non-negative")
    root = Path(source_root).resolve()
    reader = MMapMotionReader(output)
    if reader.subset != dataset:
        raise ValueError(f"cache subset {reader.subset!r} does not match {dataset!r}")
    seeds = candidate_seeds(
        root, dataset, limit_per_track_kind=source_records_per_track_kind
    )
    candidates = resolve_candidates(root, dataset, seeds, workers=workers)
    candidates = [candidate for candidate in candidates if candidate.sample_id in reader.clips]
    if not candidates:
        raise RuntimeError("benchmark source candidates are absent from the cache")

    # This setup pass discovers valid object/frame sizes and warms the same files
    # before either timed path. It is intentionally excluded from both timings.
    shapes: dict[str, tuple[str, int]] = {}
    for candidate in candidates:
        sample = _load_sample(root, candidate)
        obj = sample.objects[0]
        shapes[candidate.sample_id] = (obj.object_id, len(obj.points2d))

    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(candidates), size=samples + warmup)
    operations: list[Operation] = []
    for choice in choices.tolist():
        candidate = candidates[choice]
        object_id, total_frames = shapes[candidate.sample_id]
        max_start = max(0, total_frames - frames)
        operations.append(
            (candidate, object_id, int(rng.integers(0, max_start + 1)))
        )

    def raw_get(operation: Operation) -> dict[str, np.ndarray]:
        candidate, object_id, start = operation
        return _window(
            _load_sample(root, candidate),
            object_id,
            start=start,
            frames=frames,
            points=points,
        )

    def mmap_get(operation: Operation) -> dict[str, np.ndarray]:
        candidate, object_id, start = operation
        return reader.get_object_window(
            candidate.sample_id,
            object_id,
            start=start,
            frames=frames,
            points=points,
        )

    for operation in operations[:warmup]:
        _consume(raw_get(operation))
    for operation in operations[:warmup]:
        _consume(mmap_get(operation))
    timed = operations[warmup:]
    raw = _measure(raw_get, timed)
    mmap = _measure(mmap_get, timed)
    if not np.isclose(raw["checksum"], mmap["checksum"], rtol=1e-12, atol=1e-5):
        raise ValueError("benchmark source/cache checksums differ")
    result = {
        "status": "completed",
        "dataset": dataset,
        "measured_at": utc_now(),
        "cache_output": str(Path(output).resolve()),
        "parameters": {
            "source_records_per_track_kind": source_records_per_track_kind,
            "samples": samples,
            "warmup": warmup,
            "frames": frames,
            "points": points,
            "workers": workers,
            "seed": seed,
        },
        "raw_tar_npz": raw,
        "mmap_npy_parquet_index": mmap,
        "throughput_speedup": mmap["samples_per_second"] / raw["samples_per_second"],
        "checksums_equal": True,
        "limitations": [
            "This is a warm-cache single-process numeric-window benchmark.",
            "RGB/video decode, PyTorch workers, and GPU transfer are not measured.",
            "Setup/index construction is excluded from both timed paths.",
        ],
    }
    close_cached_archives()
    return result
