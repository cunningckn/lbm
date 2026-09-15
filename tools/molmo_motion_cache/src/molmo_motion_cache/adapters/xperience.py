"""Xperience source-schema adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import Candidate, TrackObject
from .shared import empty_camera, normalized_sample, track_object


def split_files(annotation_root: Path) -> list[tuple[str, Path]]:
    return [
        ("object", annotation_root / "xperience_split.json"),
        ("hand", annotation_root / "xperience_hand_split.json"),
    ]


def member_names(
    track_kind: str, metadata: dict[str, Any]
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    video_id = str(metadata["file"])
    if track_kind == "object":
        return (("object", f"tracks/{video_id}_object.npz"),), ()
    if track_kind == "hand":
        ranges = metadata.get("clips_by_object", {})
        if not isinstance(ranges, dict):
            raise ValueError(f"invalid Xperience hand ranges for {video_id}")
        roles = [role for role in ("left_hand", "right_hand") if role in ranges]
        if not roles:
            raise ValueError(f"Xperience hand sample has no hand role: {video_id}")
        return tuple((role, f"tracks/{video_id}_{role}.npz") for role in roles), ()
    raise ValueError(f"unsupported Xperience track kind: {track_kind}")


def load_sample(
    _source_root: str | Path,
    candidate: Candidate,
    tracks: dict[str, dict[str, np.ndarray]],
    _cameras: dict[str, Any],
):
    objects: list[TrackObject] = []
    if candidate.track_kind == "object":
        document = tracks["object"]
        objects.append(
            track_object(
                "object",
                document["tracks_2d"],
                document["points_3d"],
                document.get("visibility_2d"),
                document.get("visibility"),
                points2d_order="KT",
                points3d_order="KT",
                trust_weights=document.get("trust_weights"),
                trust_weights_order="KT",
                keep_mask=document.get("keep_mask"),
            )
        )
    elif candidate.track_kind == "hand":
        for role, document in tracks.items():
            if "object_name" in document and str(document["object_name"]) != role:
                raise ValueError(
                    f"Xperience hand role mismatch for {candidate.video_id}: "
                    f"{role!r} != {str(document['object_name'])!r}"
                )
            objects.append(
                track_object(
                    role,
                    document["pixel_coords"],
                    document["points_3d"],
                    document.get("visibility_2d"),
                    document.get("visibility"),
                    points2d_order="KT",
                    points3d_order="KT",
                )
            )
    else:
        raise ValueError(f"unsupported Xperience track kind: {candidate.track_kind}")
    camera = empty_camera("requires-upstream-xperience-reconstruction")
    return normalized_sample(
        candidate,
        dimensions=np.asarray([512, 512], dtype=np.int64),
        objects=objects,
        camera=camera,
    )
