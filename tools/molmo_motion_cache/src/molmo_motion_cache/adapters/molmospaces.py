"""MolmoSpaces source-schema adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..archives import read_npz
from .models import CameraData, Candidate, TrackObject
from .shared import (
    has_empty_track_member,
    mapping,
    normalized_sample,
    paired_track_members,
    track_object,
)


def split_files(annotation_root: Path) -> list[tuple[str, Path]]:
    return [("object", annotation_root / "molmospaces_split.json")]


def member_names(
    track_kind: str, metadata: dict[str, Any]
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if track_kind != "object":
        raise ValueError(f"unsupported MolmoSpaces track kind: {track_kind}")
    video_id = str(metadata["file"])
    return paired_track_members("tracks", video_id), (("camera", f"camera/{video_id}.npz"),)


def load_sample(
    source_root: str | Path,
    candidate: Candidate,
    tracks: dict[str, dict[str, np.ndarray]],
    cameras: dict[str, Any],
):
    d2 = tracks["track_2d"]
    dimensions = np.asarray(d2["dim"])
    points2d = mapping(d2["tracks"], "molmospaces 2D")
    visibility2d = mapping(d2["visibility"], "molmospaces 2D visibility")
    if has_empty_track_member(candidate, "track_3d"):
        # The published release has one zero-byte 3D member. Preserve its 2D
        # trajectory and c2w camera, but make unavailable 3D explicit.
        points3d = {
            key: np.full((value.shape[1], value.shape[0], 3), np.nan, dtype=np.float32)
            for key, value in points2d.items()
        }
        visibility3d = {
            key: np.zeros((value.shape[1], value.shape[0]), dtype=np.bool_)
            for key, value in points2d.items()
        }
    else:
        d3 = tracks["track_3d"]
        points3d = mapping(d3["points_3d"], "molmospaces 3D")
        visibility3d = mapping(d3["visibility"], "molmospaces 3D visibility")
    if set(points2d) != set(points3d):
        raise ValueError(f"molmospaces object keys differ for {candidate.video_id}")
    objects: list[TrackObject] = []
    for key in points2d:
        objects.append(
            track_object(
                key,
                points2d[key],
                points3d[key],
                visibility2d[key],
                visibility3d[key],
                points2d_order="TK",
                points3d_order="KT",
            )
        )
    document = read_npz(source_root, cameras["camera"])
    poses = np.asarray(document["cam_poses"], dtype=np.float32)
    intrinsics = np.asarray(document["intrinsics"], dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError(f"invalid MolmoSpaces camera for {candidate.video_id}")
    # The source stores a c2w matrix. World-frame points project after callers
    # invert it; conversion deliberately preserves this direction unchanged.
    camera: CameraData = CameraData(
        poses=np.ascontiguousarray(poses),
        pose_indices=np.arange(len(poses), dtype=np.int64),
        dynamic_intrinsics=np.empty((0, 4), dtype=np.float32),
        intrinsic_indices=np.empty((0,), dtype=np.int64),
        static_intrinsics=np.ascontiguousarray(intrinsics[None]),
        pose_convention="camera-to-world",
        availability="materialized",
    )
    return normalized_sample(candidate, dimensions=dimensions, objects=objects, camera=camera)
