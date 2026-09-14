"""DROID adapter for the portable MolmoMotion mmap cache.

The adapter intentionally consumes only completed tar files named
tracks-*.tar and camera-*.tar.  Downloader partials conventionally start with
a dot and end in .part, so they cannot match this input contract.
"""

from __future__ import annotations

import json
import os
import tempfile
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import (
    file_stat_record,
    fsync_directory,
    read_json,
    utc_now,
    write_checksum_manifest,
    write_json,
    write_parquet,
)


@dataclass(frozen=True)
class MemberRef:
    """A tar member referenced only by a path relative to the source root."""

    tar_relpath: str
    member_name: str


@dataclass(frozen=True)
class DroidCandidate:
    sample_id: str
    video_id: str
    split: str
    metadata: dict[str, Any]
    track_2d: MemberRef | None
    track_3d: MemberRef | None
    camera: MemberRef | None


@dataclass(frozen=True)
class PreparedDroidRecord:
    candidate: DroidCandidate
    num_frames: int
    num_points: int
    row_count: int
    extrinsics_kind: str


@dataclass
class DroidSample:
    candidate: DroidCandidate
    points2d: np.ndarray
    points3d: np.ndarray
    visibility2d: np.ndarray
    visibility3d: np.ndarray
    ds_dim: np.ndarray
    intrinsics_measured: np.ndarray
    intrinsics_ds: np.ndarray
    extrinsics: np.ndarray
    extrinsics_kind: str


class DroidSource:
    """Indexes completed DROID tar shards and loads individual source records."""

    def __init__(self, source_root: str | Path) -> None:
        self.root = Path(source_root).resolve()
        self.droid_root = self.root / "droid"
        if not self.droid_root.is_dir():
            raise FileNotFoundError(f"DROID source directory does not exist: {self.droid_root}")
        self._archives: dict[str, tarfile.TarFile] = {}
        self._track_2d: dict[str, MemberRef] = {}
        self._track_3d: dict[str, MemberRef] = {}
        self._cameras: dict[str, MemberRef] = {}
        self._camera_documents: dict[MemberRef, dict[str, Any]] = {}
        self._index_completed_shards()

    def __enter__(self) -> "DroidSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def _open_and_index(self, path: Path) -> tuple[str, tarfile.TarFile]:
        relative = path.relative_to(self.root).as_posix()
        archive = tarfile.open(path, mode="r:*")
        self._archives[relative] = archive
        return relative, archive

    @staticmethod
    def _record_member(
        mapping: dict[str, MemberRef],
        key: str,
        reference: MemberRef,
        kind: str,
    ) -> None:
        existing = mapping.get(key)
        if existing is not None:
            raise ValueError(
                f"duplicate {kind} member for {key}: "
                f"{existing.tar_relpath}:{existing.member_name} and "
                f"{reference.tar_relpath}:{reference.member_name}"
            )
        mapping[key] = reference

    def _index_completed_shards(self) -> None:
        tracks_dir = self.droid_root / "tracks"
        camera_dir = self.droid_root / "camera"
        track_tars = sorted(tracks_dir.glob("tracks-*.tar"))
        camera_tars = sorted(camera_dir.glob("camera-*.tar"))
        if not track_tars:
            raise FileNotFoundError(f"no completed DROID track tar was found in {tracks_dir}")
        if not camera_tars:
            raise FileNotFoundError(f"no completed DROID camera tar was found in {camera_dir}")

        for path in track_tars:
            relative, archive = self._open_and_index(path)
            for member in archive:
                if not member.isfile():
                    continue
                if member.name.startswith("tracks/") and member.name.endswith("_2d.npz"):
                    key = member.name[len("tracks/") : -len("_2d.npz")]
                    self._record_member(
                        self._track_2d, key, MemberRef(relative, member.name), "2D track"
                    )
                elif member.name.startswith("tracks/") and member.name.endswith("_3d.npz"):
                    key = member.name[len("tracks/") : -len("_3d.npz")]
                    self._record_member(
                        self._track_3d, key, MemberRef(relative, member.name), "3D track"
                    )

        for path in camera_tars:
            relative, archive = self._open_and_index(path)
            for member in archive:
                if not member.isfile():
                    continue
                if member.name.startswith("camera/") and member.name.endswith("_cameras.json"):
                    scene = member.name[len("camera/") : -len("_cameras.json")]
                    self._record_member(
                        self._cameras,
                        scene,
                        MemberRef(relative, member.name),
                        "camera",
                    )

    @property
    def source_index_counts(self) -> dict[str, int]:
        return {
            "track_2d_members": len(self._track_2d),
            "track_3d_members": len(self._track_3d),
            "camera_members": len(self._cameras),
        }

    def _load_npz(self, reference: MemberRef) -> dict[str, np.ndarray]:
        archive = self._archives[reference.tar_relpath]
        member = archive.getmember(reference.member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"cannot extract {reference.member_name}")
        with extracted:
            with np.load(extracted, allow_pickle=False) as archive_arrays:
                return {key: np.array(archive_arrays[key], copy=True) for key in archive_arrays.files}

    def _load_camera_document(self, reference: MemberRef) -> dict[str, Any]:
        cached = self._camera_documents.get(reference)
        if cached is not None:
            return cached
        archive = self._archives[reference.tar_relpath]
        member = archive.getmember(reference.member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"cannot extract {reference.member_name}")
        with extracted:
            document = json.load(extracted)
        if not isinstance(document, dict):
            raise ValueError(f"camera document is not a mapping: {reference.member_name}")
        self._camera_documents[reference] = document
        return document

    @staticmethod
    def _as_matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        return array

    @staticmethod
    def scaled_intrinsics(measured: np.ndarray, ds_dim: np.ndarray) -> np.ndarray:
        """Scale source measured K into the official DROID track image geometry."""

        height, width = (int(ds_dim[0]), int(ds_dim[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid DROID ds_dim: {ds_dim.tolist()}")
        cx, cy = float(measured[0, 2]), float(measured[1, 2])
        if cx <= 0 or cy <= 0:
            raise ValueError(f"cannot scale intrinsics with principal point {(cx, cy)}")
        affine = np.diag(
            np.array([width / (2.0 * cx), height / (2.0 * cy), 1.0], dtype=np.float32)
        )
        return (affine @ measured).astype(np.float32, copy=False)

    def candidates(self) -> list[DroidCandidate]:
        split_path = self.droid_root / "annotations" / "droid_split.json"
        split_document = read_json(split_path)
        if not isinstance(split_document, dict):
            raise ValueError(f"expected a split mapping in {split_path}")
        candidates: list[DroidCandidate] = []
        for split in ("train", "test"):
            entries = split_document.get(split, [])
            if not isinstance(entries, list):
                raise ValueError(f"DROID split {split!r} is not a list")
            for metadata in entries:
                if not isinstance(metadata, dict):
                    raise ValueError(f"DROID split {split!r} contains a non-object entry")
                video_id = metadata.get("file")
                camera_id = metadata.get("cam")
                if not isinstance(video_id, str) or not isinstance(camera_id, str):
                    raise ValueError(f"DROID entry is missing file/cam: {metadata!r}")
                scene, separator, camera_suffix = video_id.rpartition("__")
                if not separator or camera_suffix != camera_id:
                    raise ValueError(f"DROID file/camera mismatch: {video_id!r} vs {camera_id!r}")
                candidates.append(
                    DroidCandidate(
                        sample_id=f"droid/{video_id}",
                        video_id=video_id,
                        split=split,
                        metadata=metadata,
                        track_2d=self._track_2d.get(video_id),
                        track_3d=self._track_3d.get(video_id),
                        camera=self._cameras.get(scene),
                    )
                )
        return candidates

    def load(self, candidate: DroidCandidate) -> DroidSample:
        if candidate.track_2d is None or candidate.track_3d is None:
            raise FileNotFoundError(f"track pair is unavailable for {candidate.video_id}")
        if candidate.camera is None:
            raise FileNotFoundError(f"camera document is unavailable for {candidate.video_id}")
        array_2d = self._load_npz(candidate.track_2d)
        array_3d = self._load_npz(candidate.track_3d)
        required_2d = {"tracks_2d", "visibility", "ds_dim"}
        required_3d = {"points_3d", "valid_3d"}
        missing = (required_2d - array_2d.keys()) | (required_3d - array_3d.keys())
        if missing:
            raise ValueError(f"{candidate.video_id} is missing required arrays: {sorted(missing)}")

        points2d = np.asarray(array_2d["tracks_2d"])
        points3d = np.asarray(array_3d["points_3d"])
        visibility2d = np.asarray(array_2d["visibility"])
        visibility3d = np.asarray(array_3d["valid_3d"])
        ds_dim = np.asarray(array_2d["ds_dim"], dtype=np.int64)
        if points2d.ndim != 3 or points2d.shape[-1] != 2:
            raise ValueError(f"tracks_2d has invalid shape for {candidate.video_id}: {points2d.shape}")
        if points3d.ndim != 3 or points3d.shape[-1] != 3:
            raise ValueError(f"points_3d has invalid shape for {candidate.video_id}: {points3d.shape}")
        if points2d.shape[:2] != points3d.shape[:2]:
            raise ValueError(f"2D/3D shape mismatch for {candidate.video_id}")
        if visibility2d.shape != points2d.shape[:2] or visibility3d.shape != points2d.shape[:2]:
            raise ValueError(f"visibility shape mismatch for {candidate.video_id}")
        if ds_dim.shape != (2,):
            raise ValueError(f"ds_dim has invalid shape for {candidate.video_id}: {ds_dim.shape}")
        if points2d.dtype != np.float32 or points3d.dtype != np.float32:
            raise ValueError(
                f"unexpected DROID point dtype for {candidate.video_id}: "
                f"{points2d.dtype}, {points3d.dtype}"
            )

        camera_document = self._load_camera_document(candidate.camera)
        camera_id = candidate.metadata["cam"]
        calibration = camera_document.get(camera_id)
        if not isinstance(calibration, dict):
            raise ValueError(f"camera {camera_id} is absent from {candidate.camera.member_name}")
        measured = self._as_matrix(
            calibration.get("measured_intrinsics"), (3, 3), "measured_intrinsics"
        )
        if "vggt_extrinsics" in calibration:
            extrinsics_kind = "vggt_extrinsics"
            extrinsics_value = calibration["vggt_extrinsics"]
        elif "optimized_extrinsics" in calibration:
            extrinsics_kind = "optimized_extrinsics"
            extrinsics_value = calibration["optimized_extrinsics"]
        else:
            raise ValueError(f"no supported extrinsics are present for {candidate.video_id}")
        extrinsics = self._as_matrix(extrinsics_value, (4, 4), extrinsics_kind)
        intrinsics_ds = self.scaled_intrinsics(measured, ds_dim)

        expected_frames = candidate.metadata.get("num_frames")
        if expected_frames is not None and int(expected_frames) != points2d.shape[0]:
            raise ValueError(
                f"annotation/track frame mismatch for {candidate.video_id}: "
                f"{expected_frames} != {points2d.shape[0]}"
            )
        return DroidSample(
            candidate=candidate,
            points2d=points2d,
            points3d=points3d,
            visibility2d=visibility2d.astype(np.bool_, copy=False),
            visibility3d=visibility3d.astype(np.bool_, copy=False),
            ds_dim=ds_dim,
            intrinsics_measured=measured,
            intrinsics_ds=intrinsics_ds,
            extrinsics=extrinsics,
            extrinsics_kind=extrinsics_kind,
        )

    def source_file_records(self, prepared: Iterable[PreparedDroidRecord]) -> list[dict[str, Any]]:
        source_paths: set[Path] = {
            self.droid_root / "annotations" / "droid_split.json",
            self.droid_root / "annotations" / "droid_clips.json",
            self.droid_root / "annotations" / "droid_videos_index.json",
        }
        for record in prepared:
            candidate = record.candidate
            for reference in (candidate.track_2d, candidate.track_3d, candidate.camera):
                if reference is not None:
                    source_paths.add(self.root / reference.tar_relpath)
        return [file_stat_record(path, self.root) for path in sorted(source_paths)]


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _chunked(values: list[PreparedDroidRecord], size: int) -> Iterable[list[PreparedDroidRecord]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _prepare_records(
    source: DroidSource, limit: int | None
) -> tuple[list[PreparedDroidRecord], Counter[str], int]:
    selected: list[PreparedDroidRecord] = []
    skipped: Counter[str] = Counter()
    seen = 0
    for candidate in source.candidates():
        seen += 1
        if candidate.track_2d is None or candidate.track_3d is None:
            skipped["missing_track_pair"] += 1
            continue
        if candidate.camera is None:
            skipped["missing_camera_document"] += 1
            continue
        try:
            sample = source.load(candidate)
        except (OSError, ValueError, KeyError) as error:
            skipped[type(error).__name__] += 1
            continue
        num_frames, num_points = sample.points2d.shape[:2]
        selected.append(
            PreparedDroidRecord(
                candidate=candidate,
                num_frames=num_frames,
                num_points=num_points,
                row_count=num_frames * num_points,
                extrinsics_kind=sample.extrinsics_kind,
            )
        )
        if limit is not None and len(selected) >= limit:
            break
    return selected, skipped, seen


def _clip_row(record: PreparedDroidRecord, shard_relpath: str) -> dict[str, Any]:
    metadata = record.candidate.metadata
    ds_dim = metadata["ds_dim"]
    return {
        "sample_id": record.candidate.sample_id,
        "subset": "droid",
        "video_id": record.candidate.video_id,
        "split": record.candidate.split,
        "caption": str(metadata.get("caption", "")),
        "category": str(metadata.get("category", "")),
        "fps": float(metadata.get("fps", 15.0)),
        "height": int(ds_dim[0]),
        "width": int(ds_dim[1]),
        "num_frames": record.num_frames,
        "num_points": record.num_points,
        "motion_ranges_json": _compact_json(metadata.get("clips_by_object", {})),
        "shard": shard_relpath,
        "source_track_2d_tar": record.candidate.track_2d.tar_relpath,
        "source_track_2d_member": record.candidate.track_2d.member_name,
        "source_track_3d_tar": record.candidate.track_3d.tar_relpath,
        "source_track_3d_member": record.candidate.track_3d.member_name,
        "source_camera_tar": record.candidate.camera.tar_relpath,
        "source_camera_member": record.candidate.camera.member_name,
        "source_camera_id": str(metadata["cam"]),
    }


def _object_rows(record: PreparedDroidRecord) -> list[dict[str, Any]]:
    ranges = record.candidate.metadata.get("clips_by_object", {})
    if not isinstance(ranges, dict):
        raise ValueError(f"clips_by_object is not a mapping for {record.candidate.video_id}")
    rows: list[dict[str, Any]] = []
    for object_id, object_ranges in ranges.items():
        if not isinstance(object_ranges, list):
            raise ValueError(f"motion range is not a list for {record.candidate.video_id}")
        for range_index, interval in enumerate(object_ranges):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"invalid motion interval for {record.candidate.video_id}: {interval!r}")
            start, end = int(interval[0]), int(interval[1])
            if start < 0 or end < start or end >= record.num_frames:
                raise ValueError(f"motion interval is out of bounds for {record.candidate.video_id}")
            rows.append(
                {
                    "sample_id": record.candidate.sample_id,
                    "object_id": str(object_id),
                    "range_index": range_index,
                    "start_frame": start,
                    "end_frame_inclusive": end,
                }
            )
    return rows


def _write_shard(
    source: DroidSource,
    records: list[PreparedDroidRecord],
    shard_dir: Path,
    shard_relpath: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shard_dir.mkdir(parents=True, exist_ok=False)
    total_rows = sum(record.row_count for record in records)
    num_records = len(records)
    arrays = {
        "points3d": np.lib.format.open_memmap(
            shard_dir / "points3d.npy", mode="w+", dtype=np.float32, shape=(total_rows, 3)
        ),
        "points2d": np.lib.format.open_memmap(
            shard_dir / "points2d.npy", mode="w+", dtype=np.float32, shape=(total_rows, 2)
        ),
        "visibility3d": np.lib.format.open_memmap(
            shard_dir / "visibility3d.npy", mode="w+", dtype=np.bool_, shape=(total_rows,)
        ),
        "visibility2d": np.lib.format.open_memmap(
            shard_dir / "visibility2d.npy", mode="w+", dtype=np.bool_, shape=(total_rows,)
        ),
        "camera_intrinsics_measured": np.lib.format.open_memmap(
            shard_dir / "camera_intrinsics_measured.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_records, 3, 3),
        ),
        "camera_intrinsics_ds": np.lib.format.open_memmap(
            shard_dir / "camera_intrinsics_ds.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_records, 3, 3),
        ),
        "camera_extrinsics": np.lib.format.open_memmap(
            shard_dir / "camera_extrinsics.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_records, 4, 4),
        ),
    }
    tracks: list[dict[str, Any]] = []
    cameras: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    row_offset = 0
    try:
        for camera_row, record in enumerate(records):
            sample = source.load(record.candidate)
            if sample.points2d.shape[:2] != (record.num_frames, record.num_points):
                raise ValueError(f"source shape changed while writing {record.candidate.video_id}")
            row_count = record.row_count
            next_offset = row_offset + row_count
            arrays["points3d"][row_offset:next_offset] = sample.points3d.reshape(-1, 3)
            arrays["points2d"][row_offset:next_offset] = sample.points2d.reshape(-1, 2)
            arrays["visibility3d"][row_offset:next_offset] = sample.visibility3d.reshape(-1)
            arrays["visibility2d"][row_offset:next_offset] = sample.visibility2d.reshape(-1)
            arrays["camera_intrinsics_measured"][camera_row] = sample.intrinsics_measured
            arrays["camera_intrinsics_ds"][camera_row] = sample.intrinsics_ds
            arrays["camera_extrinsics"][camera_row] = sample.extrinsics
            clips.append(_clip_row(record, shard_relpath))
            tracks.append(
                {
                    "sample_id": record.candidate.sample_id,
                    "shard": shard_relpath,
                    "row_offset": row_offset,
                    "row_count": row_count,
                    "num_frames": record.num_frames,
                    "num_points": record.num_points,
                    "points2d_dtype": "float32",
                    "points3d_dtype": "float32",
                    "visibility_dtype": "bool",
                }
            )
            cameras.append(
                {
                    "sample_id": record.candidate.sample_id,
                    "shard": shard_relpath,
                    "camera_row": camera_row,
                    "intrinsics_measured_dtype": "float32",
                    "intrinsics_ds_dtype": "float32",
                    "extrinsics_dtype": "float32",
                    "extrinsics_kind": sample.extrinsics_kind,
                }
            )
            objects.extend(_object_rows(record))
            row_offset = next_offset
    finally:
        for array in arrays.values():
            array.flush()
        arrays.clear()
    if row_offset != total_rows:
        raise AssertionError(f"shard write mismatch: {row_offset} != {total_rows}")
    write_json(
        shard_dir / "shard.json",
        {
            "format": "molmo-motion-cache",
            "format_version": 1,
            "subset": "droid",
            "records": num_records,
            "trajectory_rows": total_rows,
            "paths": {
                "points3d": "points3d.npy",
                "points2d": "points2d.npy",
                "visibility3d": "visibility3d.npy",
                "visibility2d": "visibility2d.npy",
                "camera_intrinsics_measured": "camera_intrinsics_measured.npy",
                "camera_intrinsics_ds": "camera_intrinsics_ds.npy",
                "camera_extrinsics": "camera_extrinsics.npy",
            },
        },
    )
    return clips, tracks, cameras, objects


def _equal_array(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _verify_staged_output(
    staged_root: Path,
    source: DroidSource,
    prepared: list[PreparedDroidRecord],
    checks: int,
) -> dict[str, Any]:
    """Validate layout and source/output equality before the readiness marker."""

    from .reader import MMapDroidReader

    reader = MMapDroidReader(staged_root)
    expected_ids = {record.candidate.sample_id for record in prepared}
    actual_ids = set(reader.sample_ids)
    if actual_ids != expected_ids:
        raise ValueError(
            f"index sample IDs differ: expected {len(expected_ids)}, got {len(actual_ids)}"
        )
    check_count = min(max(checks, 0), len(prepared))
    checked_ids: list[str] = []
    if check_count:
        positions = np.linspace(0, len(prepared) - 1, num=check_count, dtype=np.int64)
        for position in positions.tolist():
            record = prepared[position]
            raw = source.load(record.candidate)
            cached = reader.get_full(record.candidate.sample_id)
            for name, source_array, cache_array in (
                ("points2d", raw.points2d, cached["points2d"]),
                ("points3d", raw.points3d, cached["points3d"]),
                ("visibility2d", raw.visibility2d, cached["visibility2d"]),
                ("visibility3d", raw.visibility3d, cached["visibility3d"]),
                ("intrinsics_measured", raw.intrinsics_measured, cached["intrinsics_measured"]),
                ("intrinsics_ds", raw.intrinsics_ds, cached["intrinsics_ds"]),
                ("extrinsics", raw.extrinsics, cached["extrinsics"]),
            ):
                if not _equal_array(source_array, cache_array):
                    raise ValueError(f"source/cache mismatch for {record.candidate.sample_id}: {name}")
            checked_ids.append(record.candidate.sample_id)
    return {
        "status": "passed",
        "checked_source_records": len(checked_ids),
        "checked_sample_ids": checked_ids,
        "records_indexed": len(actual_ids),
        "all_runtime_paths_relative": True,
        "verified_at": utc_now(),
    }


def build_droid_cache(
    source_root: str | Path,
    output: str | Path,
    *,
    limit: int | None,
    shard_size: int,
    checks: int,
) -> dict[str, Any]:
    """Build a DROID cache safely and return the readiness metadata.

    A positive limit makes this a pilot. Passing None processes every source
    record that is currently complete and valid.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing cache output: {output_path}; "
            "choose a new output directory or inspect the existing build"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_root = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent)
    )
    try:
        with DroidSource(source_root) as source:
            prepared, skipped, candidates_seen = _prepare_records(source, limit)
            if not prepared:
                raise RuntimeError("no complete and valid DROID samples were available to convert")
            if limit is None and skipped:
                raise RuntimeError(
                    "refusing to label an incomplete DROID conversion as full: "
                    f"{sum(skipped.values())} of {candidates_seen} canonical records failed "
                    f"validation ({dict(sorted(skipped.items()))})"
                )
            clips: list[dict[str, Any]] = []
            tracks: list[dict[str, Any]] = []
            cameras: list[dict[str, Any]] = []
            objects: list[dict[str, Any]] = []
            for shard_number, records in enumerate(_chunked(prepared, shard_size)):
                shard_name = f"{shard_number:06d}"
                shard_relpath = f"shards/droid/{shard_name}"
                shard_dir = staged_root / shard_relpath
                written = _write_shard(source, records, shard_dir, shard_relpath)
                shard_clips, shard_tracks, shard_cameras, shard_objects = written
                clips.extend(shard_clips)
                tracks.extend(shard_tracks)
                cameras.extend(shard_cameras)
                objects.extend(shard_objects)

            write_parquet(clips, staged_root / "clips.parquet")
            write_parquet(tracks, staged_root / "tracks_index.parquet")
            write_parquet(cameras, staged_root / "cameras_index.parquet")
            write_parquet(objects, staged_root / "objects.parquet")
            write_parquet(
                source.source_file_records(prepared),
                staged_root / "provenance" / "source_manifest.parquet",
            )
            total_rows = sum(record.row_count for record in prepared)
            status = "pilot" if limit is not None else "droid-complete-input"
            write_json(
                staged_root / "dataset.json",
                {
                    "format": "molmo-motion-cache",
                    "format_version": 1,
                    "subset": "droid",
                    "build_scope": status,
                    "created_at": utc_now(),
                    "records": len(prepared),
                    "trajectory_rows": total_rows,
                    "frames_present": False,
                    "frame_note": (
                        "DROID videos are reconstructed from an upstream corpus and "
                        "were not present in this conversion input."
                    ),
                    "coordinate_conventions": {
                        "points3d": "DROID exterior-camera frame; source float32, NaNs preserved",
                        "points2d": "DROID ds_dim pixel coordinates; source float32",
                        "intrinsics_measured": "source measured calibration geometry",
                        "intrinsics_ds": "measured intrinsics affine-scaled to ds_dim",
                        "extrinsics": "source vggt_extrinsics when supplied, otherwise optimized_extrinsics",
                    },
                    "time_conventions": {
                        "fps": "per-clip column in clips.parquet",
                        "frame_axis": "axis 0 of every trajectory",
                    },
                    "runtime_contract": {
                        "all_runtime_paths_relative": True,
                        "requires_raw_npz_or_tar": False,
                        "requires_raw_json": False,
                        "contains_absolute_source_paths": False,
                    },
                    "layout": {
                        "clips": "clips.parquet",
                        "tracks_index": "tracks_index.parquet",
                        "cameras_index": "cameras_index.parquet",
                        "objects": "objects.parquet",
                        "source_manifest": "provenance/source_manifest.parquet",
                        "shards": "shards/droid",
                    },
                },
            )
            write_json(
                staged_root / "build_stats.json",
                {
                    "status": status,
                    "source_index_counts": source.source_index_counts,
                    "candidates_seen": candidates_seen,
                    "records_written": len(prepared),
                    "trajectory_rows_written": total_rows,
                    "shards_written": (len(prepared) + shard_size - 1) // shard_size,
                    "skipped_before_limit": dict(sorted(skipped.items())),
                    "source_input_policy": (
                        "only completed tracks-*.tar and camera-*.tar files were read; "
                        "partial transfer files were ignored"
                    ),
                    "created_at": utc_now(),
                },
            )
            verification = _verify_staged_output(staged_root, source, prepared, checks)
            write_json(staged_root / "verification.json", verification)
        manifest_hash = write_checksum_manifest(staged_root)
        ready = {
            "format": "molmo-motion-cache",
            "format_version": 1,
            "status": "pilot-ready" if limit is not None else "ready",
            "records": len(prepared),
            "sha256sums_sha256": manifest_hash,
            "created_at": utc_now(),
        }
        marker_name = "PILOT_READY.json" if limit is not None else "READY.json"
        write_json(staged_root / marker_name, ready)
        fsync_directory(staged_root)
        os.replace(staged_root, output_path)
        fsync_directory(output_path.parent)
        return {"output": str(output_path), **ready}
    except BaseException as error:
        raise RuntimeError(
            f"DROID cache build failed; partial staging was preserved for inspection at {staged_root}"
        ) from error
