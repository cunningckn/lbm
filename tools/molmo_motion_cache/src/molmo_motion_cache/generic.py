"""Build and verify materialized MolmoMotion trajectory subset caches.

Source-schema parsing is in :mod:`molmo_motion_cache.adapters`; this module
only coordinates shard writing, output verification, and delivery metadata.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .adapters import (
    CameraData,
    Candidate,
    CandidateSeed,
    NormalizedSample,
    TrackObject,
    candidate_seeds,
    has_empty_track_member,
    load_sample,
    resolve_candidates,
    track_object,
)
from .archives import close_cached_archives
from .common import (
    GENERIC_SUBSETS,
    file_stat_record,
    fsync_directory,
    read_parquet_rows,
    utc_now,
    write_checksum_manifest,
    write_completion_marker,
    write_json,
    write_parquet,
)


# Compatibility aliases keep existing helper imports working while source parsing
# lives exclusively in adapters/.
_load_sample = load_sample
_track_object = track_object
_has_empty_track_member = has_empty_track_member

__all__ = [
    "CameraData",
    "Candidate",
    "CandidateSeed",
    "GENERIC_SUBSETS",
    "NormalizedSample",
    "TrackObject",
    "build_generic_cache",
    "candidate_seeds",
    "resolve_candidates",
    "_has_empty_track_member",
    "_load_sample",
    "_track_object",
]


@dataclass
class ShardResult:
    shard_number: int
    clips: list[dict[str, Any]]
    tracks: list[dict[str, Any]]
    cameras: list[dict[str, Any]]
    motion_ranges: list[dict[str, Any]]
    records: int
    trajectory_rows: int


def _chunks(values: list[Candidate], size: int) -> Iterable[list[Candidate]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _memmap(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _write_shard(
    source_root: str,
    staged_root: str,
    dataset: str,
    shard_number: int,
    candidates: list[Candidate],
) -> ShardResult:
    samples = [_load_sample(source_root, candidate) for candidate in candidates]
    shard_relpath = f"shards/{dataset}/{shard_number:06d}"
    shard_root = Path(staged_root) / shard_relpath
    shard_root.mkdir(parents=True, exist_ok=False)
    trajectory_rows = sum(
        int(obj.points2d.shape[0] * obj.points2d.shape[1])
        for sample in samples
        for obj in sample.objects
    )
    pose_rows = sum(len(sample.camera.poses) for sample in samples)
    dynamic_intrinsic_rows = sum(len(sample.camera.dynamic_intrinsics) for sample in samples)
    static_intrinsic_rows = sum(len(sample.camera.static_intrinsics) for sample in samples)
    arrays: dict[str, np.ndarray] = {
        "points2d": _memmap(shard_root / "points2d.npy", np.float32, (trajectory_rows, 2)),
        "points3d": _memmap(shard_root / "points3d.npy", np.float32, (trajectory_rows, 3)),
        "visibility2d": _memmap(shard_root / "visibility2d.npy", np.bool_, (trajectory_rows,)),
        "visibility3d": _memmap(shard_root / "visibility3d.npy", np.bool_, (trajectory_rows,)),
    }
    if dataset == "xperience":
        arrays["trust_weights"] = _memmap(
            shard_root / "trust_weights.npy", np.float32, (trajectory_rows,)
        )
        arrays["trust_weights"][:] = np.nan
    if pose_rows:
        arrays["camera_poses"] = _memmap(
            shard_root / "camera_poses.npy", np.float32, (pose_rows, 4, 4)
        )
        arrays["camera_pose_indices"] = _memmap(
            shard_root / "camera_pose_indices.npy", np.int64, (pose_rows,)
        )
    if dynamic_intrinsic_rows:
        arrays["camera_intrinsics_dynamic"] = _memmap(
            shard_root / "camera_intrinsics_dynamic.npy",
            np.float32,
            (dynamic_intrinsic_rows, 4),
        )
        arrays["camera_intrinsic_indices"] = _memmap(
            shard_root / "camera_intrinsic_indices.npy",
            np.int64,
            (dynamic_intrinsic_rows,),
        )
    if static_intrinsic_rows:
        arrays["camera_intrinsics_static"] = _memmap(
            shard_root / "camera_intrinsics_static.npy",
            np.float32,
            (static_intrinsic_rows, 3, 3),
        )

    clips: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    camera_rows: list[dict[str, Any]] = []
    motion_rows: list[dict[str, Any]] = []
    trajectory_offset = pose_offset = dynamic_offset = static_offset = 0
    try:
        for sample in samples:
            candidate = sample.candidate
            clips.append(
                {
                    "sample_id": candidate.sample_id,
                    "dataset": candidate.dataset,
                    "track_kind": candidate.track_kind,
                    "video_id": candidate.video_id,
                    "split": candidate.split,
                    "caption": str(candidate.metadata.get("caption", "")),
                    "fps": float(candidate.metadata.get("fps", 0.0)),
                    "height": sample.height,
                    "width": sample.width,
                    "num_frames": int(candidate.metadata["num_frames"]),
                    "num_objects": len(sample.objects),
                    "source_3d_availability": (
                        "unavailable-empty-source-member"
                        if _has_empty_track_member(candidate, "track_3d")
                        else "materialized"
                    ),
                    "shard": shard_relpath,
                }
            )
            ranges = candidate.metadata.get("clips_by_object", {})
            if not isinstance(ranges, dict):
                raise ValueError(f"invalid motion ranges for {candidate.sample_id}")
            for obj in sample.objects:
                frames, points = obj.points2d.shape[:2]
                row_count = frames * points
                next_offset = trajectory_offset + row_count
                arrays["points2d"][trajectory_offset:next_offset] = obj.points2d.reshape(-1, 2)
                arrays["points3d"][trajectory_offset:next_offset] = obj.points3d.reshape(-1, 3)
                arrays["visibility2d"][trajectory_offset:next_offset] = obj.visibility2d.reshape(-1)
                arrays["visibility3d"][trajectory_offset:next_offset] = obj.visibility3d.reshape(-1)
                if obj.trust_weights is not None:
                    arrays["trust_weights"][trajectory_offset:next_offset] = (
                        obj.trust_weights.reshape(-1)
                    )
                track_rows.append(
                    {
                        "sample_id": candidate.sample_id,
                        "object_id": obj.object_id,
                        "shard": shard_relpath,
                        "row_offset": trajectory_offset,
                        "row_count": row_count,
                        "num_frames": frames,
                        "num_points": points,
                        "trust_weights_available": obj.trust_weights is not None,
                        "keep_mask": obj.keep_mask.tolist() if obj.keep_mask is not None else None,
                        "source_point_indices": (
                            obj.source_point_indices.tolist()
                            if not np.array_equal(obj.source_point_indices, np.arange(points))
                            else None
                        ),
                    }
                )
                for range_index, interval in enumerate(ranges.get(obj.object_id, [])):
                    if not isinstance(interval, list) or len(interval) != 2:
                        raise ValueError(f"invalid motion interval for {candidate.sample_id}")
                    start, end = int(interval[0]), int(interval[1])
                    if start < 0 or end < start or end >= frames:
                        raise ValueError(f"motion interval out of bounds for {candidate.sample_id}")
                    motion_rows.append(
                        {
                            "sample_id": candidate.sample_id,
                            "object_id": obj.object_id,
                            "range_index": range_index,
                            "start_frame": start,
                            "end_frame_inclusive": end,
                        }
                    )
                trajectory_offset = next_offset

            camera = sample.camera
            pose_count = len(camera.poses)
            dynamic_count = len(camera.dynamic_intrinsics)
            static_count = len(camera.static_intrinsics)
            if pose_count:
                arrays["camera_poses"][pose_offset : pose_offset + pose_count] = camera.poses
                arrays["camera_pose_indices"][pose_offset : pose_offset + pose_count] = camera.pose_indices
            if dynamic_count:
                arrays["camera_intrinsics_dynamic"][
                    dynamic_offset : dynamic_offset + dynamic_count
                ] = camera.dynamic_intrinsics
                arrays["camera_intrinsic_indices"][
                    dynamic_offset : dynamic_offset + dynamic_count
                ] = camera.intrinsic_indices
            if static_count:
                arrays["camera_intrinsics_static"][
                    static_offset : static_offset + static_count
                ] = camera.static_intrinsics
            camera_rows.append(
                {
                    "sample_id": candidate.sample_id,
                    "shard": shard_relpath,
                    "availability": camera.availability,
                    "pose_convention": camera.pose_convention,
                    "pose_offset": pose_offset,
                    "pose_count": pose_count,
                    "dynamic_intrinsic_offset": dynamic_offset,
                    "dynamic_intrinsic_count": dynamic_count,
                    "static_intrinsic_offset": static_offset,
                    "static_intrinsic_count": static_count,
                }
            )
            pose_offset += pose_count
            dynamic_offset += dynamic_count
            static_offset += static_count
    finally:
        for array in arrays.values():
            array.flush()
        arrays.clear()

    if trajectory_offset != trajectory_rows:
        raise AssertionError("trajectory row accounting mismatch")
    write_json(
        shard_root / "shard.json",
        {
            "format": "molmo-motion-cache",
            "format_version": 1,
            "dataset": dataset,
            "records": len(samples),
            "trajectory_rows": trajectory_rows,
            "camera_pose_rows": pose_rows,
            "camera_dynamic_intrinsic_rows": dynamic_intrinsic_rows,
            "camera_static_intrinsic_rows": static_intrinsic_rows,
        },
    )
    return ShardResult(
        shard_number=shard_number,
        clips=clips,
        tracks=track_rows,
        cameras=camera_rows,
        motion_ranges=motion_rows,
        records=len(samples),
        trajectory_rows=trajectory_rows,
    )


def _source_files(root: Path, dataset: str, candidates: list[Candidate]) -> list[dict[str, Any]]:
    paths = {
        root / reference.tar_relpath
        for candidate in candidates
        for _, reference in (*candidate.track_members, *candidate.camera_members)
    }
    annotation_root = root / dataset / "annotations"
    paths.update(annotation_root.glob("*.json"))
    return [file_stat_record(path, root) for path in sorted(paths)]


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating):
        return bool(
            np.array_equal(
                left.astype(np.float32, copy=False),
                right.astype(np.float32, copy=False),
                equal_nan=True,
            )
        )
    return bool(np.array_equal(left, right))


def _verify_output(
    source_root: Path,
    staged_root: Path,
    candidates: list[Candidate],
    checks: int,
) -> dict[str, Any]:
    clip_rows = read_parquet_rows(staged_root / "clips.parquet")
    track_rows = read_parquet_rows(staged_root / "tracks_index.parquet")
    camera_rows = read_parquet_rows(staged_root / "cameras_index.parquet")
    if len(clip_rows) != len(candidates):
        raise ValueError(f"clip index count mismatch: {len(clip_rows)} != {len(candidates)}")
    tracks_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in track_rows:
        tracks_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    cameras_by_sample = {str(row["sample_id"]): row for row in camera_rows}
    count = min(max(checks, 0), len(candidates))
    positions = (
        np.linspace(0, len(candidates) - 1, num=count, dtype=np.int64).tolist() if count else []
    )
    checked: list[str] = []
    opened: dict[str, dict[str, np.ndarray]] = {}
    for position in positions:
        candidate = candidates[position]
        raw = _load_sample(source_root, candidate)
        rows = tracks_by_sample[candidate.sample_id]
        if len(rows) != len(raw.objects):
            raise ValueError(f"object count mismatch for {candidate.sample_id}")
        shard = str(rows[0]["shard"])
        arrays = opened.get(shard)
        if arrays is None:
            shard_root = staged_root / shard
            arrays = {
                name: np.load(path, mmap_mode="r")
                for name in (
                    "points2d",
                    "points3d",
                    "visibility2d",
                    "visibility3d",
                    "trust_weights",
                    "camera_poses",
                    "camera_pose_indices",
                    "camera_intrinsics_dynamic",
                    "camera_intrinsic_indices",
                    "camera_intrinsics_static",
                )
                if (path := shard_root / f"{name}.npy").is_file()
            }
            opened[shard] = arrays
        raw_by_id = {obj.object_id: obj for obj in raw.objects}
        for row in rows:
            obj = raw_by_id[str(row["object_id"])]
            start = int(row["row_offset"])
            stop = start + int(row["row_count"])
            frames, points = int(row["num_frames"]), int(row["num_points"])
            for name, expected, width in (
                ("points2d", obj.points2d, 2),
                ("points3d", obj.points3d, 3),
                ("visibility2d", obj.visibility2d, None),
                ("visibility3d", obj.visibility3d, None),
            ):
                shape = (frames, points, width) if width else (frames, points)
                actual = arrays[name][start:stop].reshape(shape)
                if not _equal(expected, actual):
                    raise ValueError(f"source/cache mismatch for {candidate.sample_id}: {name}")
            weights_available = bool(row["trust_weights_available"])
            if weights_available != (obj.trust_weights is not None):
                raise ValueError(f"trust weight availability mismatch for {candidate.sample_id}")
            if obj.trust_weights is not None:
                actual_weights = arrays["trust_weights"][start:stop].reshape(frames, points)
                if not _equal(obj.trust_weights, actual_weights):
                    raise ValueError(f"source/cache mismatch for {candidate.sample_id}: trust_weights")
            cached_mask = row.get("keep_mask")
            if obj.keep_mask is None:
                if cached_mask is not None:
                    raise ValueError(f"unexpected keep mask for {candidate.sample_id}")
            elif not _equal(obj.keep_mask, np.asarray(cached_mask, dtype=np.bool_)):
                raise ValueError(f"source/cache mismatch for {candidate.sample_id}: keep_mask")
            cached_indices = row.get("source_point_indices")
            actual_indices = (
                np.arange(points, dtype=np.int64)
                if cached_indices is None
                else np.asarray(cached_indices, dtype=np.int64)
            )
            if not _equal(obj.source_point_indices, actual_indices):
                raise ValueError(f"source/cache mismatch for {candidate.sample_id}: source_point_indices")
        camera_row = cameras_by_sample[candidate.sample_id]
        camera = raw.camera
        for name, expected, offset_key, count_key in (
            ("camera_poses", camera.poses, "pose_offset", "pose_count"),
            (
                "camera_pose_indices",
                camera.pose_indices,
                "pose_offset",
                "pose_count",
            ),
            (
                "camera_intrinsics_dynamic",
                camera.dynamic_intrinsics,
                "dynamic_intrinsic_offset",
                "dynamic_intrinsic_count",
            ),
            (
                "camera_intrinsic_indices",
                camera.intrinsic_indices,
                "dynamic_intrinsic_offset",
                "dynamic_intrinsic_count",
            ),
            (
                "camera_intrinsics_static",
                camera.static_intrinsics,
                "static_intrinsic_offset",
                "static_intrinsic_count",
            ),
        ):
            amount = int(camera_row[count_key])
            if amount == 0:
                if len(expected):
                    raise ValueError(f"missing cached camera rows for {candidate.sample_id}: {name}")
                continue
            start = int(camera_row[offset_key])
            if not _equal(expected, arrays[name][start : start + amount]):
                raise ValueError(f"source/cache mismatch for {candidate.sample_id}: {name}")
        checked.append(candidate.sample_id)
    return {
        "status": "passed",
        "records_indexed": len(clip_rows),
        "checked_source_records": len(checked),
        "checked_sample_ids": checked,
        "all_runtime_paths_relative": True,
        "verified_at": utc_now(),
    }


def build_generic_cache(
    source_root: str | Path,
    output: str | Path,
    dataset: str,
    *,
    limit_per_track_kind: int | None,
    shard_size: int,
    workers: int,
    checks: int,
    checksums: bool,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if dataset not in GENERIC_SUBSETS:
        raise ValueError(f"unsupported generic subset: {dataset}")
    if limit_per_track_kind is not None and limit_per_track_kind <= 0:
        raise ValueError("limit_per_track_kind must be positive")
    if shard_size <= 0 or workers <= 0:
        raise ValueError("shard_size and workers must be positive")
    root = Path(source_root).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing cache output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_root = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent)
    )
    try:
        seeds = candidate_seeds(root, dataset, limit_per_track_kind=limit_per_track_kind)
        candidates = resolve_candidates(root, dataset, seeds, workers=workers)
        if not candidates:
            raise RuntimeError(f"no canonical {dataset} records were available")
        chunks = list(_chunks(candidates, shard_size))
        results: list[ShardResult] = []
        pool_size = min(workers, len(chunks))
        # Do not fork conversion workers while the parent retains raw tar descriptors
        # from an earlier source check in the same process.
        close_cached_archives()
        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            futures = {
                executor.submit(
                    _write_shard,
                    str(root),
                    str(staged_root),
                    dataset,
                    shard_number,
                    chunk,
                ): shard_number
                for shard_number, chunk in enumerate(chunks)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if completed == 1 or completed % 25 == 0 or completed == len(futures):
                    print(f"[{dataset}] completed shards {completed}/{len(futures)}", flush=True)
        results.sort(key=lambda value: value.shard_number)
        clips = [row for result in results for row in result.clips]
        tracks = [row for result in results for row in result.tracks]
        cameras = [row for result in results for row in result.cameras]
        motion_ranges = [row for result in results for row in result.motion_ranges]
        write_parquet(clips, staged_root / "clips.parquet")
        write_parquet(tracks, staged_root / "tracks_index.parquet")
        write_parquet(cameras, staged_root / "cameras_index.parquet")
        write_parquet(motion_ranges, staged_root / "motion_ranges.parquet")
        write_parquet(
            _source_files(root, dataset, candidates),
            staged_root / "provenance" / "source_manifest.parquet",
        )
        source_anomalies = [
            {
                "sample_id": candidate.sample_id,
                "role": role,
                "member": reference.member_name,
                "size_bytes": reference.size,
                "handling": "NaN points3d with all-false visibility3d",
            }
            for candidate in candidates
            for role, reference in candidate.track_members
            if reference.size == 0
        ]
        scope = "pilot" if limit_per_track_kind is not None else "complete-materialized-subset"
        dataset_metadata: dict[str, Any] = {
            "format": "molmo-motion-cache",
            "format_version": 1,
            "dataset": dataset,
            "build_scope": scope,
            "records": len(candidates),
            "trajectory_rows": sum(result.trajectory_rows for result in results),
            "created_at": utc_now(),
            "runtime_contract": {
                "all_runtime_paths_relative": True,
                "requires_raw_npz_or_tar": False,
                "requires_raw_json": False,
                "contains_absolute_source_paths": False,
            },
            "normalization": {
                "track_axis_order": "T,K,D",
                "points_dtype": "float32",
                "visibility_dtype": "bool",
                "variable_object_point_counts": "flattened with Parquet offsets",
                "xperience_confidence": (
                    "trust_weights.npy plus keep_mask and explicit source_point_indices metadata"
                ),
            },
            "limitations": (
                ["Xperience camera and RGB require gated upstream reconstruction."]
                if dataset == "xperience"
                else []
            ),
            "source_anomalies": source_anomalies,
        }
        if source_identity is not None:
            dataset_metadata["source_identity"] = dict(source_identity)
        write_json(staged_root / "dataset.json", dataset_metadata)
        write_json(
            staged_root / "build_stats.json",
            {
                "status": scope,
                "records_written": len(candidates),
                "trajectory_rows_written": sum(result.trajectory_rows for result in results),
                "shards_written": len(results),
                "workers": pool_size,
                "shard_size": shard_size,
                "created_at": utc_now(),
            },
        )
        try:
            verification = _verify_output(root, staged_root, candidates, checks)
        finally:
            close_cached_archives()
        write_json(staged_root / "verification.json", verification)
        manifest_hash = write_checksum_manifest(staged_root) if checksums else None
        ready = {
            "format": "molmo-motion-cache",
            "format_version": 1,
            "status": "pilot-ready" if limit_per_track_kind is not None else "ready",
            "dataset": dataset,
            "records": len(candidates),
            "checksums": "sha256" if checksums else "disabled",
            "sha256sums_sha256": manifest_hash,
            "created_at": utc_now(),
        }
        write_completion_marker(staged_root, ready, pilot=limit_per_track_kind is not None)
        fsync_directory(staged_root)
        os.replace(staged_root, output_path)
        fsync_directory(output_path.parent)
        return {"output": str(output_path), **ready}
    except BaseException as error:
        close_cached_archives()
        raise RuntimeError(
            f"{dataset} cache build failed; partial staging was preserved at {staged_root}"
        ) from error
