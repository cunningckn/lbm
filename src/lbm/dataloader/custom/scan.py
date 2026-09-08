"""Scan a custom dataset root into lazy episode records."""

from __future__ import annotations

from pathlib import Path

from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.scan_index import cache_dir, load_scan_index, save_scan_index
from lbm.dataloader.custom.spec import CUSTOM_SPECS, CustomSpec


def scan_root(
    root: Path | str,
    spec: CustomSpec,
    *,
    max_episodes: int | None = None,
    rescan: bool = False,
) -> list[EpisodeRecord]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"custom dataset path not found: {root}")
    from lbm.dataloader.custom.common.numpy_dump import scan_numpy
    from lbm.dataloader.custom.datasets import module_for

    if not rescan:
        cached = load_scan_index(root, spec)
        if cached is not None:
            print(f"[scan] loaded {len(cached)} episodes from {cache_dir(root)}", flush=True)
            return cached[:max_episodes]
    records = module_for(spec.name).scan(root, spec, max_episodes=max_episodes)
    if not records:
        records = scan_numpy(root, spec, max_episodes=max_episodes)
    if records and max_episodes is None:
        dest = save_scan_index(root, spec, records)
        print(f"[scan] wrote {len(records)} episodes to {dest}", flush=True)
    return records


def read_vectors(record: EpisodeRecord, spec: CustomSpec, **kwargs):
    if record.kind == "numpy":
        from lbm.dataloader.custom.common.numpy_dump import read_numpy_vectors

        return read_numpy_vectors(record, spec, **kwargs)
    from lbm.dataloader.custom.datasets import module_for

    return module_for(spec.name).read_vectors(record, spec, **kwargs)


def read_frames(record: EpisodeRecord, spec: CustomSpec, cam: str, indices: list[int]):
    if record.kind == "numpy":
        from lbm.dataloader.custom.common.numpy_dump import read_numpy_frames

        return read_numpy_frames(record, spec, cam, indices)
    from lbm.dataloader.custom.datasets import module_for

    return module_for(spec.name).read_frames(record, spec, cam, indices)


def decode_custom_mmap(
    video_path: str,
    video_backend: str = "custom",
    video_backend_kwargs: dict | None = None,
    *,
    progress: bool = False,
    **_kwargs,
):
    """Whole-stream decode for JPEG mmap build. Lives here so mmap_frames stays source-only.

    ``video_backend_kwargs["mode"]`` is chosen by ``decode_kwargs``:

    - ``mp4_all``: ``video_path`` is a standalone video; decode the whole file.
    - ``lerobot``: rebuild the scanned episode from ``LerobotDump`` (parquet + dump methods).
    - ``episode``: rebuild ``EpisodeRecord`` from the kwargs and call ``read_frames``.

    Decode is native resolution. ``write_frame_cache`` letterboxes when packing JPEG.
    """
    del video_backend, _kwargs
    kw = dict(video_backend_kwargs or {})
    if "mode" not in kw:
        raise ValueError("decode_custom_mmap requires mode from decode_kwargs")
    mode = str(kw["mode"])
    if mode == "mp4_all":
        from lbm.dataloader.custom.video import read_mp4_all

        return read_mp4_all(video_path, progress=progress)
    if mode == "lerobot":
        return _decode_lerobot_mmap(kw, progress=progress)
    if mode == "episode":
        return _decode_episode_mmap(kw, progress=progress)
    raise ValueError(f"unknown mmap decode mode {mode!r}")


def source_jpegs_for_mmap(kw: dict) -> list[bytes] | None:
    """Packed JPEG/PNG bytes for mmap, if the source is already stills.

    Caller imdecodes one frame at a time and letterboxes. Do not pack the
    source bytes as-is: training ``_resize_u8`` stretches non-square stills.
    """
    mode = str(kw.get("mode") or "")
    if mode == "lerobot":
        from lbm.dataloader.custom.common.lerobot import lerobot_source_jpegs

        return lerobot_source_jpegs(kw["extra"]["lerobot"], str(kw["cam"]), _mmap_n_frames(kw))
    if mode == "episode":
        from lbm.dataloader.custom.datasets import module_for

        fn = getattr(module_for(str(kw["spec_name"])), "mmap_source_jpegs", None)
        if fn is None:
            return None
        n = _mmap_n_frames(kw)
        rec = EpisodeRecord(
            kind=str(kw["kind"]),
            path=str(kw["path"]),
            n_frames=n,
            extra=dict(kw.get("extra") or {}),
        )
        return fn(rec, CUSTOM_SPECS[str(kw["spec_name"])], str(kw["cam"]))
    return None


def _mmap_n_frames(kw: dict) -> int:
    n = int(kw.get("n_frames") or 0)
    if n <= 0:
        raise ValueError(f"mmap episode decode needs n_frames>0, got {n} for cam {kw.get('cam')!r}")
    return n


def _decode_lerobot_mmap(kw: dict, *, progress: bool = False):
    from lbm.dataloader.custom.common.lerobot import read_lerobot_frames, record_from_dump

    n = _mmap_n_frames(kw)
    rec = record_from_dump(kw["extra"]["lerobot"], n)
    spec = CUSTOM_SPECS[str(kw["spec_name"])]
    return read_lerobot_frames(rec, spec, str(kw["cam"]), list(range(n)), progress=progress)


def _decode_episode_mmap(kw: dict, *, progress: bool = False):
    del progress
    n = _mmap_n_frames(kw)
    rec = EpisodeRecord(
        kind=str(kw["kind"]),
        path=str(kw["path"]),
        n_frames=n,
        extra=dict(kw.get("extra") or {}),
    )
    return read_frames(rec, CUSTOM_SPECS[str(kw["spec_name"])], str(kw["cam"]), list(range(n)))
