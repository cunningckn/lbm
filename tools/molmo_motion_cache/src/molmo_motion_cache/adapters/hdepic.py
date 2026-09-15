"""HD-EPIC source-schema adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import Candidate, TrackObject
from .shared import (
    dynamic_camera_data,
    dynamic_camera_members,
    mapping,
    normalized_sample,
    paired_track_members,
    track_object,
)


def split_files(annotation_root: Path) -> list[tuple[str, Path]]:
    return [("object", annotation_root / "hdepic_split.json")]


def member_names(
    track_kind: str, metadata: dict[str, Any]
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if track_kind != "object":
        raise ValueError(f"unsupported HD-EPIC track kind: {track_kind}")
    video_id = str(metadata["file"])
    return paired_track_members("tracks", video_id), dynamic_camera_members(video_id)


def load_sample(
    source_root: str | Path,
    candidate: Candidate,
    tracks: dict[str, dict[str, np.ndarray]],
    cameras: dict[str, Any],
):
    d2, d3 = tracks["track_2d"], tracks["track_3d"]
    dimensions = np.asarray(d2["dim"])
    points3d = mapping(d3["points_3d"], "HD-EPIC 3D")
    visibility3d = mapping(d3["visibility"], "HD-EPIC 3D visibility")
    flat2d = np.asarray(d2["tracks"])
    flat_visibility2d = np.asarray(d2["visibility"])
    objects: list[TrackObject] = []
    offset = 0
    for key, value3d in points3d.items():
        points = int(value3d.shape[0])
        objects.append(
            track_object(
                key,
                flat2d[:, offset : offset + points],
                value3d,
                flat_visibility2d[:, offset : offset + points],
                visibility3d[key],
                points2d_order="TK",
                points3d_order="KT",
            )
        )
        offset += points
    if offset != flat2d.shape[1]:
        raise ValueError(f"HD-EPIC object blocks do not cover 2D points for {candidate.video_id}")
    camera = dynamic_camera_data(source_root, cameras, convention="world-to-camera")
    return normalized_sample(candidate, dimensions=dimensions, objects=objects, camera=camera)
