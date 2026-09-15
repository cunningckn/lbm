"""Registry and source-only loading boundary for materialized trajectory subsets."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from ..archives import TarMemberRef, build_tar_index, read_npz
from ..common import GENERIC_SUBSETS, read_json
from . import egodex, hdepic, molmospaces, xperience, ytvis
from .models import CameraData, Candidate, CandidateSeed, NormalizedSample, TrackObject
from .shared import has_empty_track_member, track_object


_ADAPTERS: dict[str, ModuleType] = {
    "egodex": egodex,
    "hdepic": hdepic,
    "molmospaces": molmospaces,
    "xperience": xperience,
    "ytvis": ytvis,
}


def _adapter(dataset: str) -> ModuleType:
    try:
        return _ADAPTERS[dataset]
    except KeyError as error:
        raise ValueError(f"unsupported generic subset: {dataset}") from error


def _split_entries(path: Path, *, limit: int | None) -> list[tuple[str, dict[str, Any]]]:
    document = read_json(path)
    rows: list[tuple[str, dict[str, Any]]] = []
    for split in ("train", "test"):
        values = document.get(split)
        if not isinstance(values, list):
            raise ValueError(f"{path} has no list-valued {split!r} split")
        selected = values if limit is None else values[:limit]
        for metadata in selected:
            if not isinstance(metadata, dict):
                raise ValueError(f"{path} has a non-object {split!r} entry")
            rows.append((split, metadata))
    return rows


def candidate_seeds(
    source_root: str | Path,
    dataset: str,
    *,
    limit_per_track_kind: int | None,
) -> list[CandidateSeed]:
    root = Path(source_root).resolve()
    adapter = _adapter(dataset)
    annotation_root = root / dataset / "annotations"
    seeds: list[CandidateSeed] = []
    seen: set[str] = set()
    for track_kind, path in adapter.split_files(annotation_root):
        for split, metadata in _split_entries(path, limit=limit_per_track_kind):
            video_id = str(metadata["file"])
            track_names, camera_names = adapter.member_names(track_kind, metadata)
            seed = CandidateSeed(
                sample_id=f"{dataset}/{track_kind}/{video_id}",
                dataset=dataset,
                track_kind=track_kind,
                video_id=video_id,
                split=split,
                metadata=metadata,
                track_member_names=track_names,
                camera_member_names=camera_names,
            )
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
    _adapter(dataset)
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


def load_sample(source_root: str | Path, candidate: Candidate) -> NormalizedSample:
    """Decode one source record and normalize it without writing cache data."""

    adapter = _adapter(candidate.dataset)
    tracks = {
        role: read_npz(source_root, reference)
        for role, reference in candidate.track_members
        if reference.size > 0
    }
    cameras = {role: reference for role, reference in candidate.camera_members}
    return adapter.load_sample(source_root, candidate, tracks, cameras)


__all__ = [
    "CameraData",
    "Candidate",
    "CandidateSeed",
    "GENERIC_SUBSETS",
    "NormalizedSample",
    "TrackObject",
    "candidate_seeds",
    "has_empty_track_member",
    "load_sample",
    "resolve_candidates",
    "track_object",
]
