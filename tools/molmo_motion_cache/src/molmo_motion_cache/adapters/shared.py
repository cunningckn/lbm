"""Small normalization helpers used by individual source adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..archives import TarMemberRef, read_npz
from .models import CameraData, Candidate, NormalizedSample, TrackObject


def paired_track_members(prefix: str, video_id: str) -> tuple[tuple[str, str], ...]:
    return (
        ("track_2d", f"{prefix}/{video_id}_2d.npz"),
        ("track_3d", f"{prefix}/{video_id}_3d.npz"),
    )


def dynamic_camera_members(video_id: str) -> tuple[tuple[str, str], ...]:
    return (
        ("camera_pose", f"camera/pose/{video_id}.npz"),
        ("camera_intrinsics", f"camera/intrinsics/{video_id}.npz"),
    )


def mapping(value: np.ndarray, label: str) -> dict[str, np.ndarray]:
    if value.shape != () or value.dtype != object:
        raise ValueError(f"{label} is not a scalar object mapping")
    result = value.item()
    if not isinstance(result, dict):
        raise ValueError(f"{label} does not contain a mapping")
    return {str(key): np.asarray(array) for key, array in result.items()}


def visibility(value: np.ndarray, *, frames: int, points: int, source_order: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if source_order == "KT":
        array = array.T
    if array.shape != (frames, points):
        raise ValueError(f"visibility shape {array.shape} != {(frames, points)}")
    return np.ascontiguousarray(array, dtype=np.bool_)


def track_object(
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
    source_point_indices: np.ndarray | None = None,
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
        else visibility(visibility2d, frames=frames, points=points, source_order=points2d_order)
    )
    v3 = (
        np.isfinite(p3).all(axis=-1)
        if visibility3d is None
        else visibility(visibility3d, frames=frames, points=points, source_order=points3d_order)
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
    if source_point_indices is None:
        indices = np.flatnonzero(mask).astype(np.int64, copy=False) if mask is not None else np.arange(points)
    else:
        indices = np.ascontiguousarray(np.asarray(source_point_indices), dtype=np.int64)
    if indices.shape != (points,) or np.any(indices < 0) or len(np.unique(indices)) != points:
        raise ValueError(
            f"invalid source point indices for {object_id}: expected {points} unique non-negative values"
        )
    return TrackObject(
        object_id=object_id,
        points2d=np.ascontiguousarray(p2, dtype=np.float32),
        points3d=np.ascontiguousarray(p3, dtype=np.float32),
        visibility2d=np.ascontiguousarray(v2, dtype=np.bool_),
        visibility3d=np.ascontiguousarray(v3, dtype=np.bool_),
        trust_weights=weights,
        keep_mask=mask,
        source_point_indices=np.ascontiguousarray(indices),
    )


def empty_camera(availability: str) -> CameraData:
    return CameraData(
        poses=np.empty((0, 4, 4), dtype=np.float32),
        pose_indices=np.empty((0,), dtype=np.int64),
        dynamic_intrinsics=np.empty((0, 4), dtype=np.float32),
        intrinsic_indices=np.empty((0,), dtype=np.int64),
        static_intrinsics=np.empty((0, 3, 3), dtype=np.float32),
        pose_convention="unavailable",
        availability=availability,
    )


def dynamic_camera_data(
    source_root: str | Path, references: dict[str, TarMemberRef], *, convention: str
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
    intr_indices = np.asarray(intr_doc.get("inds", np.arange(len(intrinsics))), dtype=np.int64)
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


def has_empty_track_member(candidate: Candidate, role: str) -> bool:
    """Return whether a required track role is a zero-byte source tar member."""

    return any(
        member_role == role and reference.size == 0 for member_role, reference in candidate.track_members
    )


def normalized_sample(
    candidate: Candidate,
    *,
    dimensions: np.ndarray,
    objects: list[TrackObject],
    camera: CameraData,
) -> NormalizedSample:
    dimensions = np.asarray(dimensions)
    if dimensions.shape != (2,):
        raise ValueError(f"invalid image dimensions for {candidate.sample_id}: {dimensions.shape}")
    expected_frames = int(candidate.metadata["num_frames"])
    if not objects or any(len(obj.points2d) != expected_frames for obj in objects):
        actual = [len(obj.points2d) for obj in objects]
        raise ValueError(
            f"track frame count differs from metadata for {candidate.sample_id}: "
            f"expected {expected_frames}, got {actual}"
        )
    return NormalizedSample(
        candidate=candidate,
        height=int(dimensions[0]),
        width=int(dimensions[1]),
        objects=objects,
        camera=camera,
    )
