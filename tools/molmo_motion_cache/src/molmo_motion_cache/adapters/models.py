"""Typed values shared by MolmoMotion source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..archives import TarMemberRef


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
    source_point_indices: np.ndarray


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
