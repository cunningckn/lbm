"""Adapters for the materialized MolmoMotion trajectory subsets.

The source schemas differ, but every adapter normalizes tracks to ``(T, K, D)``
and writes a small number of shard-level NPY arrays plus Parquet indices.  No
loose source NPZ files are created.
"""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .archives import TarMemberRef, build_tar_index, read_npz
from .common import (
    file_stat_record,
    fsync_directory,
    read_json,
    read_parquet_rows,
    utc_now,
    write_checksum_manifest,
    write_json,
    write_parquet,
)


GENERIC_SUBSETS = ("egodex", "hdepic", "molmospaces", "xperience", "ytvis")


@dataclass(frozen=True)
class CandidateSeed:
    sample_id: str
    dataset: str
    track_kind: str
    video_id: str
    split: str
    metadata: dict[str, Any]
    track_member_names: tuple[tuple[str, str], ...]
    camera_member_names: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    dataset: str
    track_kind: str
    video_id: str
    split: str
    metadata: dict[str, Any]
    track_members: tuple[tuple[str, TarMemberRef], ...]
    camera_members: tuple[tuple[str, TarMemberRef], ...]


@dataclass
class TrackObject:
    object_id: str
    points2d: np.ndarray
    points3d: np.ndarray
    visibility2d: np.ndarray
    visibility3d: np.ndarray
    trust_weights: np.ndarray | None
    keep_mask: np.ndarray | None


@dataclass
class CameraData:
    poses: np.ndarray
    pose_indices: np.ndarray
    dynamic_intrinsics: np.ndarray
    intrinsic_indices: np.ndarray
    static_intrinsics: np.ndarray
    pose_convention: str
    availability: str


@dataclass
class NormalizedSample:
    candidate: Candidate
    height: int
    width: int
    objects: list[TrackObject]
    camera: CameraData


@dataclass
class ShardResult:
    shard_number: int
    clips: list[dict[str, Any]]
    tracks: list[dict[str, Any]]
    cameras: list[dict[str, Any]]
    motion_ranges: list[dict[str, Any]]
    records: int
    trajectory_rows: int


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _split_entries(path: Path, *, limit: int | None) -> list[tuple[str, dict[str, Any]]]:
    document = read_json(path)
    rows: list[tuple[str, dict[str, Any]]] = []
    for split in ("train", "test"):
        values = document.get(split)
        if not isinstance(values, list):
            raise ValueError(f"{path} has no list-valued {split!r} split")
        selected = values if limit is None else values[:limit]
        rows.extend((split, value) for value in selected)
    return rows


def _pair(prefix: str, video_id: str) -> tuple[tuple[str, str], ...]:
    return (
        ("track_2d", f"{prefix}/{video_id}_2d.npz"),
        ("track_3d", f"{prefix}/{video_id}_3d.npz"),
    )


def _dynamic_camera(video_id: str) -> tuple[tuple[str, str], ...]:
    return (
        ("camera_pose", f"camera/pose/{video_id}.npz"),
        ("camera_intrinsics", f"camera/intrinsics/{video_id}.npz"),
    )


def _seed_from_entry(
    dataset: str,
    track_kind: str,
    split: str,
    metadata: dict[str, Any],
) -> CandidateSeed:
    video_id = str(metadata["file"])
    if dataset == "egodex":
        track_names = _pair(f"tracks/{track_kind}", video_id)
        camera_names = _dynamic_camera(video_id)
    elif dataset in {"hdepic", "ytvis"}:
        track_names = _pair("tracks", video_id)
        camera_names = _dynamic_camera(video_id)
    elif dataset == "molmospaces":
        track_names = _pair("tracks", video_id)
        camera_names = (("camera", f"camera/{video_id}.npz"),)
    elif dataset == "xperience" and track_kind == "object":
        track_names = (("object", f"tracks/{video_id}_object.npz"),)
        camera_names = ()
    elif dataset == "xperience" and track_kind == "hand":
        ranges = metadata.get("clips_by_object", {})
        if not isinstance(ranges, dict):
            raise ValueError(f"invalid Xperience hand ranges for {video_id}")
        roles = [role for role in ("left_hand", "right_hand") if role in ranges]
        if not roles:
            raise ValueError(f"Xperience hand sample has no hand role: {video_id}")
        track_names = tuple((role, f"tracks/{video_id}_{role}.npz") for role in roles)
        camera_names = ()
    else:
        raise ValueError(f"unsupported materialized subset: {dataset}/{track_kind}")
    return CandidateSeed(
        sample_id=f"{dataset}/{track_kind}/{video_id}",
        dataset=dataset,
        track_kind=track_kind,
        video_id=video_id,
        split=split,
        metadata=metadata,
        track_member_names=track_names,
        camera_member_names=camera_names,
    )


def candidate_seeds(
    source_root: str | Path,
    dataset: str,
    *,
    limit_per_track_kind: int | None,
) -> list[CandidateSeed]:
    root = Path(source_root).resolve()
    annotation_root = root / dataset / "annotations"
    groups: list[tuple[str, Path]]
    if dataset == "egodex":
        groups = [
            ("object", annotation_root / "egodex_split.json"),
            ("hand", annotation_root / "egodex_hand_split.json"),
        ]
    elif dataset == "xperience":
        groups = [
            ("object", annotation_root / "xperience_split.json"),
            ("hand", annotation_root / "xperience_hand_split.json"),
        ]
    elif dataset in {"hdepic", "molmospaces", "ytvis"}:
        groups = [("object", annotation_root / f"{dataset}_split.json")]
    else:
        raise ValueError(f"unsupported generic subset: {dataset}")

    seeds: list[CandidateSeed] = []
    seen: set[str] = set()
    for track_kind, path in groups:
        for split, metadata in _split_entries(path, limit=limit_per_track_kind):
            seed = _seed_from_entry(dataset, track_kind, split, metadata)
            if seed.sample_id in seen:
                raise ValueError(f"duplicate canonical sample ID: {seed.sample_id}")
            seen.add(seed.sample_id)
            seeds.append(seed)
    return seeds


def resolve_candidates(
    source_root: str | Path,
    dataset: str,
    seeds: list[CandidateSeed],
    *,
    workers: int,
) -> list[Candidate]:
    root = Path(source_root).resolve()
    subset_root = root / dataset
    track_names = {name for seed in seeds for _, name in seed.track_member_names}
    camera_names = {name for seed in seeds for _, name in seed.camera_member_names}
    track_wanted = track_names if len(track_names) <= 20_000 else None
    camera_wanted = camera_names if len(camera_names) <= 20_000 else None
    track_index = build_tar_index(
        root,
        (subset_root / "tracks").glob("tracks-*.tar"),
        workers=workers,
        wanted_names=track_wanted,
    )
    camera_index: dict[str, TarMemberRef] = {}
    if camera_names:
        camera_index = build_tar_index(
            root,
            (subset_root / "camera").glob("camera-*.tar"),
            workers=workers,
            wanted_names=camera_wanted,
        )

    missing: list[str] = []
    candidates: list[Candidate] = []
    for seed in seeds:
        track_members: list[tuple[str, TarMemberRef]] = []
        camera_members: list[tuple[str, TarMemberRef]] = []
        for role, name in seed.track_member_names:
            reference = track_index.get(name)
            if reference is None:
                missing.append(name)
            else:
                track_members.append((role, reference))
        for role, name in seed.camera_member_names:
            reference = camera_index.get(name)
            if reference is None:
                missing.append(name)
            else:
                camera_members.append((role, reference))
        if len(track_members) == len(seed.track_member_names) and len(camera_members) == len(
            seed.camera_member_names
        ):
            candidates.append(
                Candidate(
                    sample_id=seed.sample_id,
                    dataset=seed.dataset,
                    track_kind=seed.track_kind,
                    video_id=seed.video_id,
                    split=seed.split,
                    metadata=seed.metadata,
                    track_members=tuple(track_members),
                    camera_members=tuple(camera_members),
                )
            )
    if missing:
        preview = ", ".join(sorted(missing)[:8])
        raise FileNotFoundError(
            f"{dataset}: {len(missing)} canonical members are missing; first entries: {preview}"
        )
    return candidates


def _mapping(value: np.ndarray, label: str) -> dict[str, np.ndarray]:
    if value.shape != () or value.dtype != object:
        raise ValueError(f"{label} is not a scalar object mapping")
    result = value.item()
    if not isinstance(result, dict):
        raise ValueError(f"{label} does not contain a mapping")
    return {str(key): np.asarray(array) for key, array in result.items()}


def _visibility(value: np.ndarray, *, frames: int, points: int, source_order: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if source_order == "KT":
        array = array.T
    if array.shape != (frames, points):
        raise ValueError(f"visibility shape {array.shape} != {(frames, points)}")
    return np.ascontiguousarray(array, dtype=np.bool_)


def _track_object(
    object_id: str,
    points2d: np.ndarray,
    points3d: np.ndarray,
    visibility2d: np.ndarray | None,
    visibility3d: np.ndarray | None,
    *,
    points2d_order: str,
    points3d_order: str,
    trust_weights: np.ndarray | None = None,
    trust_weights_order: str = "TK",
    keep_mask: np.ndarray | None = None,
) -> TrackObject:
    p2 = np.asarray(points2d)
    p3 = np.asarray(points3d)
    if points2d_order == "KT":
        p2 = p2.transpose(1, 0, 2)
    if points3d_order == "KT":
        p3 = p3.transpose(1, 0, 2)
    if p2.ndim != 3 or p2.shape[-1] != 2:
        raise ValueError(f"invalid 2D track shape for {object_id}: {p2.shape}")
    if p3.ndim != 3 or p3.shape[-1] != 3:
        raise ValueError(f"invalid 3D track shape for {object_id}: {p3.shape}")
    if p2.shape[:2] != p3.shape[:2]:
        raise ValueError(f"2D/3D track mismatch for {object_id}: {p2.shape} vs {p3.shape}")
    frames, points = p2.shape[:2]
    v2 = (
        np.isfinite(p2).all(axis=-1)
        if visibility2d is None
        else _visibility(visibility2d, frames=frames, points=points, source_order=points2d_order)
    )
    v3 = (
        np.isfinite(p3).all(axis=-1)
        if visibility3d is None
        else _visibility(visibility3d, frames=frames, points=points, source_order=points3d_order)
    )
    weights: np.ndarray | None = None
    if trust_weights is not None:
        weights = np.asarray(trust_weights)
        if trust_weights_order == "KT":
            weights = weights.T
        if weights.shape != (frames, points):
            raise ValueError(
                f"trust weight shape {weights.shape} != {(frames, points)} for {object_id}"
            )
        weights = np.ascontiguousarray(weights, dtype=np.float32)
    mask: np.ndarray | None = None
    if keep_mask is not None:
        mask = np.ascontiguousarray(np.asarray(keep_mask), dtype=np.bool_)
        if mask.ndim != 1 or int(mask.sum()) != points:
            raise ValueError(
                f"keep mask does not select {points} points for {object_id}: {mask.shape}"
            )
    return TrackObject(
        object_id=object_id,
        points2d=np.ascontiguousarray(p2, dtype=np.float32),
        points3d=np.ascontiguousarray(p3, dtype=np.float32),
        visibility2d=np.ascontiguousarray(v2, dtype=np.bool_),
        visibility3d=np.ascontiguousarray(v3, dtype=np.bool_),
        trust_weights=weights,
        keep_mask=mask,
    )


def _empty_camera(availability: str) -> CameraData:
    return CameraData(
        poses=np.empty((0, 4, 4), dtype=np.float32),
        pose_indices=np.empty((0,), dtype=np.int64),
        dynamic_intrinsics=np.empty((0, 4), dtype=np.float32),
        intrinsic_indices=np.empty((0,), dtype=np.int64),
        static_intrinsics=np.empty((0, 3, 3), dtype=np.float32),
        pose_convention="unavailable",
        availability=availability,
    )


def _dynamic_camera_data(
    source_root: str | Path,
    references: dict[str, TarMemberRef],
    *,
    convention: str,
) -> CameraData:
    pose_doc = read_npz(source_root, references["camera_pose"])
    intr_doc = read_npz(source_root, references["camera_intrinsics"])
    poses = np.asarray(pose_doc["data"], dtype=np.float32)
    intrinsics = np.asarray(intr_doc["data"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"invalid dynamic camera pose shape: {poses.shape}")
    if intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
        raise ValueError(f"invalid dynamic intrinsics shape: {intrinsics.shape}")
    pose_indices = np.asarray(pose_doc.get("inds", np.arange(len(poses))), dtype=np.int64)
    intr_indices = np.asarray(
        intr_doc.get("inds", np.arange(len(intrinsics))), dtype=np.int64
    )
    if pose_indices.shape != (len(poses),) or intr_indices.shape != (len(intrinsics),):
        raise ValueError("camera data/indices length mismatch")
    return CameraData(
        poses=np.ascontiguousarray(poses),
        pose_indices=np.ascontiguousarray(pose_indices),
        dynamic_intrinsics=np.ascontiguousarray(intrinsics),
        intrinsic_indices=np.ascontiguousarray(intr_indices),
        static_intrinsics=np.empty((0, 3, 3), dtype=np.float32),
        pose_convention=convention,
        availability="materialized",
    )


def _load_sample(source_root: str | Path, candidate: Candidate) -> NormalizedSample:
    tracks = {role: read_npz(source_root, ref) for role, ref in candidate.track_members}
    cameras = {role: ref for role, ref in candidate.camera_members}
    objects: list[TrackObject] = []
    dim = np.asarray([512, 512], dtype=np.int64)

    if candidate.dataset == "egodex" and candidate.track_kind == "object":
        d2, d3 = tracks["track_2d"], tracks["track_3d"]
        dim = np.asarray(d2["dim"])
        objects.append(
            _track_object(
                "object",
                d2["tracks"],
                d3["points_3d"],
                d2["visibility"],
                d3.get("visibility"),
                points2d_order="TK",
                points3d_order="KT",
            )
        )
    elif candidate.dataset == "egodex" and candidate.track_kind == "hand":
        d2, d3 = tracks["track_2d"], tracks["track_3d"]
        dim = np.asarray(d2["dim"])
        p2, v2 = _mapping(d2["tracks"], "EgoDex hand 2D"), _mapping(
            d2["visibility"], "EgoDex hand visibility"
        )
        p3 = _mapping(d3["points_3d"], "EgoDex hand 3D")
        v3 = _mapping(d3["visibility"], "EgoDex hand 3D visibility") if "visibility" in d3 else {}
        if set(p2) != set(p3):
            raise ValueError(f"EgoDex hand object keys differ for {candidate.video_id}")
        for key in p2:
            objects.append(
                _track_object(
                    key,
                    p2[key],
                    p3[key],
                    v2.get(key),
                    v3.get(key),
                    points2d_order="TK",
                    points3d_order="KT",
                )
            )
    elif candidate.dataset == "hdepic":
        d2, d3 = tracks["track_2d"], tracks["track_3d"]
        dim = np.asarray(d2["dim"])
        p3 = _mapping(d3["points_3d"], "HD-EPIC 3D")
        v3 = _mapping(d3["visibility"], "HD-EPIC 3D visibility")
        flat2 = np.asarray(d2["tracks"])
        flat_v2 = np.asarray(d2["visibility"])
        offset = 0
        for key, value3 in p3.items():
            points = int(value3.shape[0])
            objects.append(
                _track_object(
                    key,
                    flat2[:, offset : offset + points],
                    value3,
                    flat_v2[:, offset : offset + points],
                    v3[key],
                    points2d_order="TK",
                    points3d_order="KT",
                )
            )
            offset += points
        if offset != flat2.shape[1]:
            raise ValueError(f"HD-EPIC object blocks do not cover 2D points for {candidate.video_id}")
    elif candidate.dataset in {"molmospaces", "ytvis"}:
        d2, d3 = tracks["track_2d"], tracks["track_3d"]
        dim = np.asarray(d2["dim"])
        p2 = _mapping(d2["tracks"], f"{candidate.dataset} 2D")
        v2 = _mapping(d2["visibility"], f"{candidate.dataset} 2D visibility")
        p3 = _mapping(d3["points_3d"], f"{candidate.dataset} 3D")
        v3 = _mapping(d3["visibility"], f"{candidate.dataset} 3D visibility")
        if set(p2) != set(p3):
            raise ValueError(f"{candidate.dataset} object keys differ for {candidate.video_id}")
        for key in p2:
            objects.append(
                _track_object(
                    key,
                    p2[key],
                    p3[key],
                    v2[key],
                    v3[key],
                    points2d_order="TK",
                    points3d_order="KT",
                )
            )
    elif candidate.dataset == "xperience" and candidate.track_kind == "object":
        doc = tracks["object"]
        objects.append(
            _track_object(
                "object",
                doc["tracks_2d"],
                doc["points_3d"],
                doc.get("visibility_2d"),
                doc.get("visibility"),
                points2d_order="KT",
                points3d_order="KT",
                trust_weights=doc.get("trust_weights"),
                trust_weights_order="KT",
                keep_mask=doc.get("keep_mask"),
            )
        )
    elif candidate.dataset == "xperience" and candidate.track_kind == "hand":
        for role, doc in tracks.items():
            if "object_name" in doc and str(doc["object_name"]) != role:
                raise ValueError(
                    f"Xperience hand role mismatch for {candidate.video_id}: "
                    f"{role!r} != {str(doc['object_name'])!r}"
                )
            objects.append(
                _track_object(
                    role,
                    doc["pixel_coords"],
                    doc["points_3d"],
                    doc.get("visibility_2d"),
                    doc.get("visibility"),
                    points2d_order="KT",
                    points3d_order="KT",
                )
            )
    else:
        raise AssertionError(f"unhandled adapter: {candidate.dataset}/{candidate.track_kind}")

    if dim.shape != (2,):
        raise ValueError(f"invalid image dimensions for {candidate.sample_id}: {dim.shape}")
    expected_frames = int(candidate.metadata["num_frames"])
    if not objects or any(len(obj.points2d) != expected_frames for obj in objects):
        actual = [len(obj.points2d) for obj in objects]
        raise ValueError(
            f"track frame count differs from metadata for {candidate.sample_id}: "
            f"expected {expected_frames}, got {actual}"
        )

    if candidate.dataset in {"egodex", "hdepic", "ytvis"}:
        convention = "camera-to-world" if candidate.dataset in {"egodex", "ytvis"} else "world-to-camera"
        camera = _dynamic_camera_data(source_root, cameras, convention=convention)
    elif candidate.dataset == "molmospaces":
        doc = read_npz(source_root, cameras["camera"])
        poses = np.asarray(doc["cam_poses"], dtype=np.float32)
        intrinsics = np.asarray(doc["intrinsics"], dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or intrinsics.shape != (3, 3):
            raise ValueError(f"invalid MolmoSpaces camera for {candidate.video_id}")
        camera = CameraData(
            poses=np.ascontiguousarray(poses),
            pose_indices=np.arange(len(poses), dtype=np.int64),
            dynamic_intrinsics=np.empty((0, 4), dtype=np.float32),
            intrinsic_indices=np.empty((0,), dtype=np.int64),
            static_intrinsics=np.ascontiguousarray(intrinsics[None]),
            pose_convention="camera-to-world",
            availability="materialized",
        )
    else:
        camera = _empty_camera("requires-upstream-xperience-reconstruction")

    return NormalizedSample(
        candidate=candidate,
        height=int(dim[0]),
        width=int(dim[1]),
        objects=objects,
        camera=camera,
    )


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
        scope = "pilot" if limit_per_track_kind is not None else "complete-materialized-subset"
        write_json(
            staged_root / "dataset.json",
            {
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
                    "xperience_confidence": "trust_weights.npy plus keep_mask list metadata",
                },
                "limitations": (
                    ["Xperience camera and RGB require gated upstream reconstruction."]
                    if dataset == "xperience"
                    else []
                ),
            },
        )
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
        verification = _verify_output(root, staged_root, candidates, checks)
        write_json(staged_root / "verification.json", verification)
        manifest_hash = write_checksum_manifest(staged_root) if checksums else None
        marker_name = "PILOT_READY.json" if limit_per_track_kind is not None else "READY.json"
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
        write_json(staged_root / marker_name, ready)
        fsync_directory(staged_root)
        os.replace(staged_root, output_path)
        fsync_directory(output_path.parent)
        return {"output": str(output_path), **ready}
    except BaseException as error:
        raise RuntimeError(
            f"{dataset} cache build failed; partial staging was preserved at {staged_root}"
        ) from error
