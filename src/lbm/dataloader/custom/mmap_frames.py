"""Resolve a camera stream into a JPEG mmap cache job.

Non-LeRobot dumps cache under the dump root (``dataset.root``). LeRobot keeps
one store per nested repo. Missing cameras are skipped here; the loader fills
black frames and ``camera_mask`` instead of copying another view.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import CustomSpec

_STREAM_KINDS = {"zarr", "lance", "mcap", "mp4"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".webm", ".mov"}


@dataclass(frozen=True)
class MmapFrameRef:
    """One camera stream that ``MmapFrameStore`` can cache."""

    path: Path
    video_key: str
    trajectory_id: int
    cache_root: Path
    kind: str
    cam: str
    decode_mode: str
    n_frames: int = 0
    extra: dict = field(default_factory=dict)


def _traj_id(record: EpisodeRecord) -> int:
    from lbm.dataloader.custom.common.lerobot import lerobot_of

    dump = lerobot_of(record)
    if dump is not None:
        return int(dump.episode_index)
    extra = record.extra or {}
    raw = str(Path(record.path).resolve())
    epi = extra.get("episode_index")
    if epi is not None:
        raw = f"{raw}#{int(epi)}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


def _dir_of(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def camera_has_source(record: EpisodeRecord, spec: CustomSpec, cam: str) -> bool:
    """True when ``cam`` has its own on-disk stream. Absent views are not filled from others."""
    extra = record.extra or {}
    videos = extra.get("videos")
    if isinstance(videos, dict):
        raw = videos.get(cam)
        return bool(raw) and Path(str(raw)).exists()
    keys = extra.get("video_keys")
    if isinstance(keys, dict):
        return keys.get(cam) not in (None, "")
    if extra.get("video_key") is not None:
        return cam == spec.camera_keys[0]
    from lbm.dataloader.custom.common.lerobot import lerobot_of

    dump = lerobot_of(record)
    if dump is not None:
        if dump.resolve_mp4(cam) is not None:
            return True
        from lbm.dataloader.custom.common.lerobot import _parquet_cam_column

        if dump.parquet.is_file() and _parquet_cam_column(dump.parquet, cam) is not None:
            return True
        return dump.cam_dtype(cam) == "image"
    if not record.path:
        return False
    return Path(record.path).exists()


def _video_key(cam: str, extra: dict, spec: CustomSpec | None = None, *, video_key: str | None = None) -> str:
    if video_key is not None:
        return str(video_key)
    keys = extra.get("video_keys")
    mapped = keys.get(cam) if isinstance(keys, dict) else None
    if mapped not in (None, ""):
        return str(mapped)
    if extra.get("video_key") is not None and (spec is None or cam == spec.camera_keys[0]):
        return str(extra["video_key"])
    return cam


def _cache_root(
    path: Path,
    extra: dict,
    *,
    cache_root: Path | None = None,
    dump_root: Path | str | None = None,
) -> Path:
    """Dump root for native dumps; explicit override; else the source's parent dir."""
    if dump_root is not None:
        return Path(dump_root)
    if cache_root is not None:
        return Path(cache_root)
    extra_root = extra.get("cache_root")
    if extra_root:
        return Path(extra_root)
    return _dir_of(path)


def _ref_for_path(
    record: EpisodeRecord,
    cam: str,
    path: Path,
    *,
    decode_mode: str,
    kind: str | None = None,
    video_key: str | None = None,
    extra: dict | None = None,
    cache_root: Path | None = None,
    dump_root: Path | str | None = None,
    trajectory_id: int | None = None,
    spec: CustomSpec | None = None,
) -> MmapFrameRef:
    extra = dict(extra if extra is not None else record.extra or {})
    return MmapFrameRef(
        path=path,
        video_key=_video_key(cam, extra, spec, video_key=video_key),
        trajectory_id=int(trajectory_id) if trajectory_id is not None else _traj_id(record),
        cache_root=_cache_root(path, extra, cache_root=cache_root, dump_root=dump_root),
        kind=str(kind if kind is not None else record.kind),
        cam=cam,
        decode_mode=decode_mode,
        n_frames=int(record.n_frames),
        extra=extra,
    )


def unique_frame_refs(refs: list[MmapFrameRef]) -> list[MmapFrameRef]:
    """One encode per ``(cache_root, trajectory_id, video_key)``."""
    out: list[MmapFrameRef] = []
    seen: set[tuple[str, int, str]] = set()
    for ref in refs:
        key = (str(ref.cache_root), int(ref.trajectory_id), str(ref.video_key))
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def collect_frame_refs(
    records: list[EpisodeRecord],
    spec: CustomSpec,
    *,
    dump_root: Path | str | None = None,
) -> list[MmapFrameRef]:
    """Resolve every ``(record, camera)`` stream. Metadata I/O only (threaded)."""
    from lbm.dataloader.custom.common.fs import map_threads

    cams = spec.camera_keys

    def _one(record: EpisodeRecord) -> list[MmapFrameRef]:
        tid = _traj_id(record)
        out: list[MmapFrameRef] = []
        for cam in cams:
            ref = resolve_mmap_frames(record, spec, cam, trajectory_id=tid, dump_root=dump_root)
            if ref is not None:
                out.append(ref)
        return out

    chunks = map_threads(_one, list(records), desc=f"mmap scan {spec.name}")
    return unique_frame_refs([ref for chunk in chunks for ref in chunk])


def resolve_mmap_frames(
    record: EpisodeRecord,
    spec: CustomSpec,
    cam: str,
    *,
    trajectory_id: int | None = None,
    dump_root: Path | str | None = None,
) -> MmapFrameRef | None:
    """Locate the on-disk source for ``cam``, or None if this view is absent.

    Dump-specific paths belong in each dataset's ``scan`` via ``extra["videos"]``.
    """
    kind = str(getattr(record, "kind", "") or "")
    if kind == "numpy" or not camera_has_source(record, spec, cam):
        return None
    tid = int(trajectory_id) if trajectory_id is not None else _traj_id(record)
    extra = dict(record.extra or {})
    if kind.startswith("lerobot"):
        return _resolve_lerobot(record, cam, kind, trajectory_id=tid)
    videos = extra.get("videos")
    raw = videos.get(cam) if isinstance(videos, dict) else None
    if raw:
        path = Path(raw)
        if path.exists():
            mode = "mp4_all" if path.suffix.lower() in _VIDEO_SUFFIXES else "episode"
            kind_out = "mp4" if mode == "mp4_all" else kind
            return _ref_for_path(
                record,
                cam,
                path,
                decode_mode=mode,
                kind=kind_out,
                extra=extra,
                dump_root=dump_root,
                trajectory_id=tid,
                spec=spec,
            )
    if kind in _STREAM_KINDS:
        path = Path(record.path)
        if path.exists():
            mode = "mp4_all" if kind == "mp4" else "episode"
            return _ref_for_path(
                record,
                cam,
                path,
                decode_mode=mode,
                kind=kind,
                extra=extra,
                dump_root=dump_root,
                trajectory_id=tid,
                spec=spec,
            )
    return None


def _resolve_lerobot(record: EpisodeRecord, cam: str, kind: str, *, trajectory_id: int) -> MmapFrameRef | None:
    from lbm.dataloader.custom.common.lerobot import lerobot_of

    dump = lerobot_of(record)
    if dump is None:
        return None
    extra = {"lerobot": dump}
    dtype = dump.cam_dtype(cam)
    if dtype == "video":
        return _lerobot_mp4_ref(record, cam, kind, dump, extra, trajectory_id)
    if dtype == "image":
        return _lerobot_parquet_ref(record, cam, kind, dump, extra, trajectory_id)
    return _lerobot_mp4_ref(record, cam, kind, dump, extra, trajectory_id) or _lerobot_parquet_ref(
        record, cam, kind, dump, extra, trajectory_id
    )


def _lerobot_mp4_ref(record, cam, kind, dump, extra, trajectory_id: int) -> MmapFrameRef | None:
    found = dump.resolve_mp4(cam)
    if found is None:
        return None
    path, video_key = found
    return _ref_for_path(
        record,
        cam,
        path,
        decode_mode="lerobot",
        kind=kind,
        video_key=video_key,
        extra=extra,
        cache_root=dump.repo,
        trajectory_id=trajectory_id,
    )


def _lerobot_parquet_ref(record, cam, kind, dump, extra, trajectory_id: int) -> MmapFrameRef | None:
    if not dump.parquet.is_file():
        return None
    return _ref_for_path(
        record,
        cam,
        dump.parquet,
        decode_mode="lerobot",
        kind=kind,
        extra=extra,
        cache_root=dump.repo,
        trajectory_id=trajectory_id,
    )


def decode_kwargs(ref: MmapFrameRef, spec: CustomSpec) -> dict:
    """Serialize ``ref.decode_mode`` for ``decode_custom_mmap``."""
    mode = ref.decode_mode
    if mode == "mp4_all":
        return {"mode": "mp4_all"}
    if mode == "lerobot":
        return {
            "mode": "lerobot",
            "spec_name": spec.name,
            "cam": ref.cam,
            "n_frames": int(ref.n_frames),
            "extra": {"lerobot": ref.extra["lerobot"]},
        }
    if mode == "episode":
        return {
            "mode": "episode",
            "kind": ref.kind,
            "spec_name": spec.name,
            "cam": ref.cam,
            "n_frames": int(ref.n_frames),
            "path": str(ref.path),
            "extra": dict(ref.extra),
        }
    raise ValueError(f"unknown mmap decode_mode {mode!r}")
