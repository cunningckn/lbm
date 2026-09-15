"""YT-VIS source-schema adapter."""

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
    return [("object", annotation_root / "ytvis_split.json")]


def member_names(
    track_kind: str, metadata: dict[str, Any]
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if track_kind != "object":
        raise ValueError(f"unsupported YT-VIS track kind: {track_kind}")
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
    points2d = mapping(d2["tracks"], "ytvis 2D")
    visibility2d = mapping(d2["visibility"], "ytvis 2D visibility")
    points3d = mapping(d3["points_3d"], "ytvis 3D")
    visibility3d = mapping(d3["visibility"], "ytvis 3D visibility")
    if set(points2d) != set(points3d):
        raise ValueError(f"ytvis object keys differ for {candidate.video_id}")
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
    camera = dynamic_camera_data(source_root, cameras, convention="camera-to-world")
    return normalized_sample(candidate, dimensions=dimensions, objects=objects, camera=camera)
