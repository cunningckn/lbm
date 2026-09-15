"""EgoDex source-schema adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import CameraData, Candidate, TrackObject
from .shared import (
    dynamic_camera_data,
    dynamic_camera_members,
    mapping,
    normalized_sample,
    paired_track_members,
    track_object,
)


def split_files(annotation_root: Path) -> list[tuple[str, Path]]:
    return [
        ("object", annotation_root / "egodex_split.json"),
        ("hand", annotation_root / "egodex_hand_split.json"),
    ]


def member_names(
    track_kind: str, metadata: dict[str, Any]
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    video_id = str(metadata["file"])
    if track_kind not in {"object", "hand"}:
        raise ValueError(f"unsupported EgoDex track kind: {track_kind}")
    return paired_track_members(f"tracks/{track_kind}", video_id), dynamic_camera_members(video_id)


def load_sample(
    source_root: str | Path,
    candidate: Candidate,
    tracks: dict[str, dict[str, np.ndarray]],
    cameras: dict[str, Any],
):
    d2, d3 = tracks["track_2d"], tracks["track_3d"]
    dimensions = np.asarray(d2["dim"])
    objects: list[TrackObject] = []
    if candidate.track_kind == "object":
        objects.append(
            track_object(
                "object",
                d2["tracks"],
                d3["points_3d"],
                d2["visibility"],
                d3.get("visibility"),
                points2d_order="TK",
                points3d_order="KT",
            )
        )
    elif candidate.track_kind == "hand":
        points2d = mapping(d2["tracks"], "EgoDex hand 2D")
        visibility2d = mapping(d2["visibility"], "EgoDex hand visibility")
        points3d = mapping(d3["points_3d"], "EgoDex hand 3D")
        visibility3d = mapping(d3["visibility"], "EgoDex hand 3D visibility") if "visibility" in d3 else {}
        if set(points2d) != set(points3d):
            raise ValueError(f"EgoDex hand object keys differ for {candidate.video_id}")
        for key in points2d:
            objects.append(
                track_object(
                    key,
                    points2d[key],
                    points3d[key],
                    visibility2d.get(key),
                    visibility3d.get(key),
                    points2d_order="TK",
                    points3d_order="KT",
                )
            )
    else:
        raise ValueError(f"unsupported EgoDex track kind: {candidate.track_kind}")
    camera: CameraData = dynamic_camera_data(source_root, cameras, convention="camera-to-world")
    return normalized_sample(candidate, dimensions=dimensions, objects=objects, camera=camera)
