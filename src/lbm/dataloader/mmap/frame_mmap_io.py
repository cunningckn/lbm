"""JPEG-packed mmap frame cache.

Hot path: letterbox to ``image_size`` (224) at pack time, JPEG quality 85,
``frames.bin`` + offset/length tables, mmap + parallel cv2 decode into RGB.

Decode is always native resolution. Pack streams one frame at a time
(letterbox + JPEG) so a long episode never sits in RAM as ``T×H×W``.
A rebuild unlinks ``manifest.json`` first and writes it last, so a crash
cannot leave a matching manifest on a half-written blob.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import weakref
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lbm.dataloader.mmap.jpeg_io import (
    decode_jpegs_into,
    decode_still_rgb,
    decode_thread_init,
    decode_threads_for_loaders,
    encode_jpeg_rgb,
    resize_with_pad,
)
from lbm.dataloader.mmap.mmap_io import (
    MMAP_DIRNAME,
    atomic_save_npy,
    atomic_write_text,
    cache_files_present,
    exclusive_cache_lock,
    read_manifest,
    read_source_manifest,
)
from lbm.utils.mem import reclaim_if_over, set_worker_rss_limit, worker_rss_limit_bytes

FRAMES_SUBDIR = "frames"
MANIFEST = "manifest.json"
JPEG_FILES = ("frames.bin", "offset.npy", "length.npy", MANIFEST)
LAYOUT_JPEG = "jpeg_pack"


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict | None = None,
    resize_size: tuple[int, int] | None = None,
    *,
    progress: bool = False,
) -> np.ndarray:
    """Decode a whole video for cache build (PyAV first, OpenCV fallback)."""
    del video_backend, video_backend_kwargs
    from lbm.dataloader.custom.video import read_mp4_all

    frames = read_mp4_all(video_path, progress=progress)
    if resize_size is not None and frames.shape[0]:
        frames = _resize_thwc(frames, int(resize_size[0]), int(resize_size[1]))
    return frames


def _resize_thwc(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    try:
        import cv2
    except ImportError:
        return frames
    out = np.empty((frames.shape[0], height, width, frames.shape[-1]), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return out


def _decode_from_job(job: dict):
    mod = job.get("decode_module")
    name = job.get("decode_name")
    if not mod or not name:
        return None
    import importlib

    return getattr(importlib.import_module(str(mod)), str(name))


def _run_bounded(pool, tasks, *, limit: int):
    """Keep at most ``limit`` payloads/futures queued, including completed work."""
    iterator = iter(tasks)
    pending = set()
    try:
        exhausted = False
        while pending or not exhausted:
            while len(pending) < limit and not exhausted:
                task = next(iterator, None)
                if task is None:
                    exhausted = True
                else:
                    fn, payload = task
                    pending.add(pool.submit(fn, payload))
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
    finally:
        for future in pending:
            future.cancel()


def _default_decode_workers() -> int:
    return min(8, os.cpu_count() or 8)


def _show_progress() -> bool:
    from lbm.utils.progress import progress_enabled

    return progress_enabled()


_OPEN_STORES: weakref.WeakSet[MmapFrameStore] = weakref.WeakSet()


def _abandon_inherited_pools() -> None:
    """Drop ThreadPoolExecutors copied by fork; their worker threads did not survive."""
    for store in list(_OPEN_STORES):
        store._abandon_decode_pool()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_abandon_inherited_pools)


def _parse_image_size(value: object) -> int | None:
    if value in (None, 0, False, "", "none", "None"):
        return None
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _validated_jpeg_tables(cache_dir: Path, manifest: dict):
    """Check indexes in bounded chunks before exposing any payload slices."""
    n = int(manifest["num_frames"])
    if manifest.get("layout") != LAYOUT_JPEG or n <= 0:
        raise ValueError("invalid JPEG layout or frame count")
    if min(int(manifest["height"]), int(manifest["width"])) <= 0:
        raise ValueError("invalid JPEG dimensions")
    offset = length = None
    try:
        offset = np.load(cache_dir / "offset.npy", mmap_mode="r", allow_pickle=False)
        length = np.load(cache_dir / "length.npy", mmap_mode="r", allow_pickle=False)
        if offset.shape != (n,) or length.shape != (n,):
            raise ValueError("JPEG index length differs from manifest")
        if offset.dtype != np.dtype("int64") or length.dtype != np.dtype("uint32"):
            raise ValueError("invalid JPEG index dtype")
        size = (cache_dir / "frames.bin").stat().st_size
        end = 0
        for lo in range(0, n, 65536):
            hi = min(n, lo + 65536)
            off, count = offset[lo:hi], length[lo:hi]
            if (int(off[0]) != end or np.any(count == 0)
                    or np.any(off < 0) or np.any(off > size)
                    or np.any(off[1:] != off[:-1] + count[:-1])):
                raise ValueError("invalid JPEG offsets or lengths")
            end = int(off[-1]) + int(count[-1])
        if end != size:
            raise ValueError("JPEG payload size differs from index")
        return offset, length
    except BaseException:
        for arr in (offset, length):
            mapping = getattr(arr, "_mmap", None)
            if mapping is not None:
                mapping.close()
        raise


def _jpeg_manifest_ready(cache_dir: Path, source_tag: str) -> bool:
    manifest = read_source_manifest(cache_dir, source_tag)
    if not manifest:
        return False
    try:
        tables = _validated_jpeg_tables(cache_dir, manifest)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError):
        return False
    for arr in tables:
        arr._mmap.close()
    return True


def _cache_is_ready(cache_dir: Path, source_tag: str) -> bool:
    return cache_files_present(cache_dir, JPEG_FILES) and _jpeg_manifest_ready(cache_dir, source_tag)


def align_frames_to_parquet_steps(
    frames: np.ndarray,
    timestamps: np.ndarray | None,
    from_timestamp: float = 0.0,
) -> np.ndarray:
    """One cache slot per parquet step. Training indexes slots with ``step_indices``."""
    if timestamps is None or len(frames) == 0:
        return frames
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1) + float(from_timestamp)
    n_video = int(frames.shape[0])
    if ts.size == 0 or ts.size == n_video:
        return frames
    if ts.size == 1:
        return np.ascontiguousarray(frames[:1])
    dt = float(np.median(np.diff(ts)))
    fps = (1.0 / dt) if dt > 1e-9 else 30.0
    idx = np.clip(np.rint(ts * fps).astype(np.int64), 0, n_video - 1)
    return np.ascontiguousarray(frames[idx])


def pack_jpeg_frames(jpegs: list[bytes]) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Pack per-frame JPEG bytes into a single blob with offset/length tables."""
    length = np.fromiter((len(blob) for blob in jpegs), dtype=np.uint32, count=len(jpegs))
    offset = np.zeros(len(jpegs), dtype=np.int64)
    if len(jpegs) > 1:
        np.cumsum(length[:-1], out=offset[1:])
    return b"".join(jpegs), offset, length


def _drop_manifest(cache_dir: Path) -> None:
    """Unlink first so a crash mid-rebuild cannot look like a ready cache."""
    try:
        (cache_dir / MANIFEST).unlink()
    except FileNotFoundError:
        return


def _encode_rgb_jpeg(
    frame: np.ndarray, *, jpeg_quality: int, image_size: int | None
) -> tuple[bytes, tuple[int, int]]:
    rgb = np.asarray(frame, dtype=np.uint8)
    if image_size is not None:
        rgb = resize_with_pad(rgb, int(image_size))
    return encode_jpeg_rgb(rgb, quality=jpeg_quality), (int(rgb.shape[0]), int(rgb.shape[1]))


class _JpegSpool:
    """Disk-backed encoded frames for overlapping spans in a shared video."""

    def __init__(self):
        self.file = None
        self.lengths: list[int] = []

    def append(self, blob: bytes, *, directory: Path) -> None:
        if self.file is None:
            directory.mkdir(parents=True, exist_ok=True)
            self.file = tempfile.TemporaryFile(dir=directory)
        self.file.write(blob)
        self.lengths.append(len(blob))

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        if self.file is not None:
            self.file.seek(0)
            for length in self.lengths:
                yield self.file.read(length)

    def clear(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        self.lengths.clear()


def _commit_jpeg_pack(
    cache_dir: Path,
    jpegs: Iterable[bytes],
    *,
    source_tag: str,
    jpeg_quality: int,
    image_size: int | None,
    height: int,
    width: int,
) -> None:
    """Stream payload to disk; publish the manifest only after all files exist.

    Memory is one JPEG plus the small offset/length index, not the full blob.
    The caller holds the cache lock when concurrent builders are possible.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    _drop_manifest(cache_dir)
    # A stable temporary name is overwritten on retry after a killed worker.
    tmp = cache_dir / "frames.bin.tmp"
    offsets: list[int] = []
    lengths: list[int] = []
    pos = 0
    try:
        with tmp.open("wb") as out:
            for blob in jpegs:
                if not 0 < len(blob) <= np.iinfo(np.uint32).max:
                    raise ValueError("invalid encoded frame length")
                offsets.append(pos)
                lengths.append(len(blob))
                out.write(blob)
                pos += len(blob)
            if not lengths:
                raise ValueError("write_frame_cache got no frames")
            out.flush()
            os.fsync(out.fileno())
        tmp.replace(cache_dir / "frames.bin")
        atomic_save_npy(cache_dir / "offset.npy", np.asarray(offsets, dtype=np.int64))
        atomic_save_npy(cache_dir / "length.npy", np.asarray(lengths, dtype=np.uint32))
        atomic_write_text(
            cache_dir / MANIFEST,
            json.dumps(
                {
                    "source": source_tag,
                    "layout": LAYOUT_JPEG,
                    "num_frames": len(lengths),
                    "jpeg_quality": int(jpeg_quality),
                    "height": int(height),
                    "width": int(width),
                    "image_size": int(image_size) if image_size is not None else None,
                },
                indent=2,
            ),
        )
    finally:
        tmp.unlink(missing_ok=True)


def _iter_rgb_source(
    frames: np.ndarray | None,
    source_jpegs: Iterable[bytes] | None,
    frame_iter: Iterable[np.ndarray] | None,
) -> Iterator[np.ndarray]:
    if source_jpegs is not None:
        for blob in source_jpegs:
            yield decode_still_rgb(blob)
        return
    if frame_iter is not None:
        yield from frame_iter
        return
    if frames is None:
        raise ValueError("write_frame_cache needs frames, frame_iter, or source_jpegs")
    yield from np.ascontiguousarray(frames, dtype=np.uint8)


def write_frame_cache(
    cache_dir: Path,
    *,
    source_tag: str,
    frames: np.ndarray | None = None,
    source_jpegs: Iterable[bytes] | None = None,
    frame_iter: Iterable[np.ndarray] | None = None,
    jpeg_quality: int = 85,
    image_size: int | None = None,
    desc: str | None = None,
    progress: bool | None = None,
) -> None:
    """Letterbox + JPEG each frame as it arrives. Manifest is written last."""
    items: Iterable[np.ndarray] = _iter_rgb_source(frames, source_jpegs, frame_iter)
    if progress is None:
        progress = _show_progress()
    if progress:
        from lbm.utils.progress import track

        items = track(items, desc=desc or "mmap-frames jpeg", unit="f", leave=False)
    iterator = iter(items)
    try:
        first = next(iterator, None)
        if first is None:
            raise ValueError("write_frame_cache got no frames")
        first_blob, hw = _encode_rgb_jpeg(first, jpeg_quality=jpeg_quality, image_size=image_size)
        del first

        def encoded():
            yield first_blob
            for i, frame in enumerate(iterator, start=1):
                blob, _ = _encode_rgb_jpeg(frame, jpeg_quality=jpeg_quality, image_size=image_size)
                yield blob
                if i % 64 == 0:
                    reclaim_if_over()

        _commit_jpeg_pack(
            cache_dir,
            encoded(),
            source_tag=source_tag,
            jpeg_quality=jpeg_quality,
            image_size=image_size,
            height=hw[0],
            width=hw[1],
        )
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    reclaim_if_over()


@dataclass
class MmapFrameEpisode:
    cache_dir: Path
    num_frames: int
    height: int
    width: int
    _blob: np.memmap
    _offset: np.ndarray
    _length: np.ndarray

    @classmethod
    def open(cls, cache_dir: Path) -> MmapFrameEpisode:
        manifest = read_manifest(cache_dir / MANIFEST)
        if manifest is None:
            raise FileNotFoundError(f"invalid frame mmap manifest in {cache_dir}")
        offset, length = _validated_jpeg_tables(cache_dir, manifest)
        try:
            blob = np.memmap(cache_dir / "frames.bin", dtype=np.uint8, mode="r")
        except BaseException:
            offset._mmap.close()
            length._mmap.close()
            raise
        return cls(
            cache_dir=cache_dir,
            num_frames=int(manifest["num_frames"]),
            height=int(manifest["height"]),
            width=int(manifest["width"]),
            _blob=blob,
            _offset=offset,
            _length=length,
        )

    def close(self) -> None:
        for arr in (self._blob, self._offset, self._length):
            mapping = getattr(arr, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def _safe_indices(self, frame_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if self.num_frames <= 0:
            return idx
        return np.clip(idx, 0, self.num_frames - 1)

    def gather_blobs(self, frame_indices: np.ndarray) -> list[memoryview]:
        blob = self._blob
        out: list[memoryview] = []
        for i in self._safe_indices(frame_indices):
            off = int(self._offset[i])
            ln = int(self._length[i])
            out.append(memoryview(blob)[off : off + ln])
        return out

    def get_frames(
        self,
        frame_indices: np.ndarray,
        *,
        decode_pool: ThreadPoolExecutor | None = None,
    ) -> np.ndarray:
        blobs = self.gather_blobs(frame_indices)
        packed = np.empty((len(blobs), self.height, self.width, 3), dtype=np.uint8)
        decode_jpegs_into(blobs, packed, decode_pool)
        return packed


def default_prebuild_workers() -> int:
    return max(1, min(32, os.cpu_count() or 8))


def split_pending_frame_jobs(
    grouped: dict[str, tuple[MmapFrameStore, list[dict]]],
    *,
    desc: str = "mmap-frames ready",
) -> tuple[dict[str, tuple[MmapFrameStore, list[dict]]], int]:
    """One ready-pass over every store. Missing ``cache_root`` dirs are all-pending."""
    from lbm.dataloader.custom.common.fs import map_ready, map_threads

    stores = [(key, store, jobs) for key, (store, jobs) in grouped.items()]
    if not stores:
        return {}, 0

    root_ok = map_threads(lambda item: item[1].cache_root.is_dir(), stores)
    if not any(root_ok):
        return grouped, 0

    has_root = {id(store): ok for (_, store, _), ok in zip(stores, root_ok, strict=True)}
    pairs = [(store, job) for _key, store, jobs in stores for job in jobs]

    def _hit(pair: tuple[MmapFrameStore, dict]) -> bool:
        store, job = pair
        return bool(has_root.get(id(store))) and store.job_ready(job)

    pending, ready = map_ready(_hit, pairs, desc=desc)
    out: dict[str, tuple[MmapFrameStore, list[dict]]] = {}
    for store, job in pending:
        out.setdefault(str(store.dataset_path), (store, []))[1].append(job)
    return out, len(ready)


def _lerobot_mp4_span(kw: dict, *, v3_only: bool = False) -> tuple[int, int] | None:
    """Contiguous packed-file ``[start, stop)``. None if the cam is not a linear mp4 span."""
    if str(kw.get("mode") or "") != "lerobot":
        return None
    dump = (kw.get("extra") or {}).get("lerobot")
    cam = kw.get("cam")
    n = int(kw.get("n_frames") or 0)
    if dump is None or cam is None or n <= 0:
        return None
    if v3_only and not getattr(dump, "is_v3", False):
        return None
    idxs = dump.mp4_indices(str(cam), list(range(n)))
    from lbm.dataloader.custom.video import contiguous_span

    return contiguous_span(idxs)


def lerobot_v3_file_span(job: dict) -> tuple[int, int] | None:
    """Packed-file ``[start, stop)`` from v3 ``from_timestamp``. None if not shareable."""
    return _lerobot_mp4_span(job.get("video_backend_kwargs") or {}, v3_only=True)


def partition_shared_frame_jobs(jobs: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    """Jobs that share one packed mp4 vs one-file-per-episode jobs."""
    by_path: dict[str, list[dict]] = {}
    for job in jobs:
        by_path.setdefault(str(job["video_path"]), []).append(job)
    shared: list[list[dict]] = []
    solo: list[dict] = []
    for group in by_path.values():
        if len(group) > 1 and all(lerobot_v3_file_span(j) is not None for j in group):
            shared.append(group)
        else:
            solo.extend(group)
    return shared, solo


def _source_jpegs_of(decode_fn, kwargs: dict) -> Iterable[bytes] | None:
    if decode_fn is None:
        return None
    import importlib

    getter = getattr(importlib.import_module(decode_fn.__module__), "source_jpegs_for_mmap", None)
    if getter is None:
        return None
    return getter(kwargs)


def _decode_stacked_rgb(
    video_path: Path,
    *,
    video_backend: str,
    video_backend_kwargs: dict,
    decode_all_frames,
    progress: bool,
) -> np.ndarray:
    decode = decode_all_frames or get_all_frames
    return decode(
        video_path.as_posix(),
        video_backend=video_backend,
        video_backend_kwargs=video_backend_kwargs,
        progress=progress,
    )


def _iter_native_rgb(video_path: Path, kwargs: dict, *, progress: bool = False):
    """Stream native RGB for a contiguous mp4. None means the caller should stack-decode."""
    from lbm.dataloader.custom.video import iter_mp4_all, iter_mp4_span

    mode = str((kwargs or {}).get("mode") or "")
    if mode == "lerobot":
        span = _lerobot_mp4_span(kwargs)
        if span is None:
            return None
        return iter_mp4_span(video_path, span[0], span[1], progress=progress)
    if mode in ("", "mp4_all"):
        return iter_mp4_all(video_path, progress=progress)
    return None


def _rgb_for_cache(
    video_path: Path,
    *,
    video_backend: str,
    video_backend_kwargs: dict,
    episode_timestamps: np.ndarray | None,
    from_timestamp: float,
    decode_all_frames,
    progress: bool,
) -> tuple[np.ndarray | None, Iterator[np.ndarray] | None]:
    """Exactly one of ``(frames, iterator)`` is set. Prefer streaming a contiguous mp4."""
    stacked = dict(
        video_backend=video_backend,
        video_backend_kwargs=video_backend_kwargs,
        decode_all_frames=decode_all_frames,
        progress=progress,
    )
    if episode_timestamps is not None:
        frames = _decode_stacked_rgb(video_path, **stacked)
        return align_frames_to_parquet_steps(frames, episode_timestamps, from_timestamp), None

    stream = _iter_native_rgb(video_path, video_backend_kwargs, progress=progress)
    if stream is None:
        return _decode_stacked_rgb(video_path, **stacked), None

    first = next(iter(stream), None)
    if first is not None:

        def _rest():
            yield first
            yield from stream

        return None, _rest()
    if decode_all_frames is None:
        return _decode_stacked_rgb(video_path, **stacked), None
    return np.zeros((0, 1, 1, 3), dtype=np.uint8), None


def _write_encoded_job(
    store: MmapFrameStore,
    job: dict,
    jpegs: Iterable[bytes],
    hw: tuple[int, int],
) -> None:
    cache_dir = store.cache_dir_for(int(job["trajectory_id"]), str(job["video_key"]))
    source_tag = store._source_tag(
        Path(job["video_path"]),
        job.get("episode_timestamps"),
        float(job["from_timestamp"]),
    )
    with exclusive_cache_lock(cache_dir / ".lock"):
        if _cache_is_ready(cache_dir, source_tag):
            return
        _commit_jpeg_pack(
            cache_dir,
            jpegs,
            source_tag=source_tag,
            jpeg_quality=store.jpeg_quality,
            image_size=store.image_size,
            height=hw[0],
            width=hw[1],
        )


@dataclass
class _SharedSpan:
    job: dict
    start: int
    stop: int
    jpegs: _JpegSpool = field(default_factory=_JpegSpool)
    hw: tuple[int, int] | None = None

    def accept(self, file_i: int, frame: np.ndarray, store: MmapFrameStore) -> bool:
        """Encode this file index if it belongs here. False once the span is written."""
        if not (self.start <= file_i < self.stop):
            return True
        blob, self.hw = _encode_rgb_jpeg(
            frame, jpeg_quality=store.jpeg_quality, image_size=store.image_size
        )
        self.jpegs.append(blob, directory=store.dataset_path / MMAP_DIRNAME)
        if len(self.jpegs) < self.stop - self.start:
            return True
        assert self.hw is not None
        _write_encoded_job(store, self.job, self.jpegs, self.hw)
        self.jpegs.clear()
        reclaim_if_over()
        return False


def write_shared_mp4_caches(store: MmapFrameStore, jobs: list[dict], *, progress: bool = False) -> None:
    """Decode one packed mp4 once; JPEG-encode each frame without stacking the file."""
    from lbm.dataloader.custom.video import iter_mp4_span

    spans: list[_SharedSpan] = []
    for job in jobs:
        span = lerobot_v3_file_span(job)
        if span is None:
            _prebuild_frame_job({**job, "progress": progress})
            continue
        spans.append(_SharedSpan(job, span[0], span[1]))
    if not spans:
        return
    spans.sort(key=lambda item: item.start)
    path = Path(spans[0].job["video_path"])
    file_i = spans[0].start
    max_stop = max(item.stop for item in spans)
    pending = list(spans)
    stream = iter_mp4_span(path, file_i, max_stop, progress=progress)
    try:
        for frame in stream:
            pending = [item for item in pending if item.accept(file_i, frame, store)]
            file_i += 1
            if not pending:
                break
        for item in pending:
            item.jpegs.clear()
            _prebuild_frame_job({**item.job, "progress": progress})
    finally:
        stream.close()
        for item in spans:
            item.jpegs.clear()


def _shared_payload(
    store: MmapFrameStore, jobs: list[dict], *, progress: bool = False, rss_limit: int | None = None
) -> dict:
    return {
        "dataset_path": str(store.dataset_path),
        "jpeg_quality": int(store.jpeg_quality),
        "image_size": store.image_size,
        "jobs": jobs,
        "progress": progress,
        "video_path": jobs[0]["video_path"],
        "rss_limit": rss_limit,
    }


def _bind_worker_rss_limit(payload: dict) -> None:
    limit = payload.get("rss_limit")
    if limit is None:
        return
    set_worker_rss_limit(int(limit))


def _open_worker_store(payload: dict, decode_job: dict) -> MmapFrameStore:
    _bind_worker_rss_limit(payload)
    return MmapFrameStore(
        payload["dataset_path"],
        jpeg_quality=int(payload["jpeg_quality"]),
        image_size=payload["image_size"],
        decode_workers=1,
        decode_all_frames=_decode_from_job(decode_job),
    )


@contextmanager
def _worker_store(payload: dict, decode_job: dict):
    store = _open_worker_store(payload, decode_job)
    try:
        yield store
    finally:
        store.close()
        reclaim_if_over()


def _prebuild_shared_file_job(payload: dict) -> str:
    jobs = payload["jobs"]
    with _worker_store(payload, jobs[0]) as store:
        write_shared_mp4_caches(store, jobs, progress=bool(payload.get("progress")))
    return str(payload["video_path"])


def _prebuild_frame_job(job: dict) -> str:
    """Spawn-safe worker: decode one episode×camera into the JPEG mmap cache."""
    timestamps = job.get("episode_timestamps")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
    with _worker_store(job, job) as store:
        store._load_or_build(
            trajectory_id=int(job["trajectory_id"]),
            video_key=str(job["video_key"]),
            video_path=Path(job["video_path"]),
            episode_timestamps=timestamps,
            from_timestamp=float(job["from_timestamp"]),
            video_backend=str(job["video_backend"]),
            video_backend_kwargs=job.get("video_backend_kwargs") or {},
            progress=bool(job.get("progress")),
        )
    return f"{job['trajectory_id']}:{job['video_key']}"


class MmapFrameStore:
    """Build-once JPEG pack cache for episode camera streams."""

    def __init__(
        self,
        dataset_path: Path | str,
        *,
        enabled: bool = True,
        jpeg_quality: int = 85,
        decode_workers: int | None = None,
        image_size: object = 224,
        decode_all_frames=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.enabled = enabled
        self.jpeg_quality = int(jpeg_quality)
        self.image_size = _parse_image_size(image_size)
        self._decode_all_frames = decode_all_frames
        n_workers = _default_decode_workers() if decode_workers is None else int(decode_workers)
        self.decode_workers = max(1, n_workers)
        self._decode_workers_auto = decode_workers is None
        self.cache_root = self.dataset_path / MMAP_DIRNAME / FRAMES_SUBDIR
        self._episodes: dict[str, MmapFrameEpisode] = {}
        self._decode_pool: ThreadPoolExecutor | None = None
        self._decode_pool_pid: int | None = None
        self.allow_build = True
        _OPEN_STORES.add(self)

    def tune_decode_workers_for_loaders(self, n_loaders: int) -> None:
        if not self._decode_workers_auto:
            return
        n = decode_threads_for_loaders(n_loaders)
        if n == self.decode_workers:
            return
        self.decode_workers = n
        self._abandon_decode_pool()

    def _abandon_decode_pool(self) -> None:
        self._decode_pool = None
        self._decode_pool_pid = None

    def _get_decode_pool(self) -> ThreadPoolExecutor | None:
        if self.decode_workers <= 1:
            return None
        pid = os.getpid()
        if self._decode_pool is not None and self._decode_pool_pid == pid:
            return self._decode_pool
        self._decode_pool = ThreadPoolExecutor(
            max_workers=self.decode_workers,
            initializer=decode_thread_init,
            thread_name_prefix="lbm-jpeg",
        )
        self._decode_pool_pid = pid
        return self._decode_pool

    def close(self) -> None:
        pool, self._decode_pool = self._decode_pool, None
        self._decode_pool_pid = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        episodes, self._episodes = self._episodes, {}
        for episode in episodes.values():
            episode.close()

    def cache_dir_for(self, trajectory_id: int, video_key: str) -> Path:
        safe_key = video_key.replace("/", "__")
        return self.cache_root / f"episode_{int(trajectory_id):06d}" / safe_key

    def make_job(
        self,
        *,
        trajectory_id: int,
        video_key: str,
        video_path: Path | str,
        episode_timestamps: np.ndarray | None,
        from_timestamp: float,
        video_backend: str,
        video_backend_kwargs: dict | None,
    ) -> dict:
        timestamps = None
        if episode_timestamps is not None:
            timestamps = np.asarray(episode_timestamps, dtype=np.float64).reshape(-1).copy()
        job = {
            "dataset_path": str(self.dataset_path),
            "jpeg_quality": int(self.jpeg_quality),
            "image_size": self.image_size,
            "trajectory_id": int(trajectory_id),
            "video_key": str(video_key),
            "video_path": str(Path(video_path)),
            "episode_timestamps": timestamps,
            "from_timestamp": float(from_timestamp),
            "video_backend": str(video_backend),
            "video_backend_kwargs": dict(video_backend_kwargs or {}),
        }
        fn = self._decode_all_frames
        if fn is not None:
            job["decode_module"] = getattr(fn, "__module__", "")
            job["decode_name"] = getattr(fn, "__name__", "")
        return job

    def job_ready(self, job: dict) -> bool:
        cache_dir = self.cache_dir_for(int(job["trajectory_id"]), str(job["video_key"]))
        if not cache_files_present(cache_dir, JPEG_FILES):
            return False
        try:
            tag = self._source_tag(
                Path(job["video_path"]),
                job.get("episode_timestamps"),
                float(job["from_timestamp"]),
            )
        except OSError:
            return False
        return _jpeg_manifest_ready(cache_dir, tag)

    def prebuild(self, jobs: list[dict], *, workers: int = 0, check_ready: bool = True) -> tuple[int, int]:
        if not jobs:
            return 0, 0
        if not check_ready:
            pending, skipped = jobs, 0
        elif not self.cache_root.is_dir():
            pending, skipped = jobs, 0
        else:
            from lbm.dataloader.custom.common.fs import map_ready

            pending, ready = map_ready(self.job_ready, jobs, desc="mmap-frames ready")
            skipped = len(ready)
        if not pending:
            from lbm.utils.progress import progress_enabled

            if skipped and progress_enabled():
                print(f"mmap-frames: {skipped} already cached", flush=True)
            return 0, skipped
        shared, solo = partition_shared_frame_jobs(pending)
        n_units = len(shared) + len(solo)
        nproc = workers if workers and workers > 0 else default_prebuild_workers()
        nproc = max(1, min(int(nproc), n_units))
        rss_limit = worker_rss_limit_bytes(nproc)
        set_worker_rss_limit(rss_limit)
        from lbm.utils.progress import track

        if nproc == 1:
            for group in shared:
                write_shared_mp4_caches(self, group, progress=True)
                reclaim_if_over()
            for job in track(solo, desc="prebuild mmap-frames", unit="vid") if solo else ():
                _prebuild_frame_job({**job, "progress": True, "rss_limit": rss_limit})
            return len(pending), skipped
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as pool:
            def tasks():
                for group in shared:
                    yield _prebuild_shared_file_job, _shared_payload(self, group, rss_limit=rss_limit)
                for job in solo:
                    yield _prebuild_frame_job, {**job, "rss_limit": rss_limit}

            for _ in track(
                _run_bounded(pool, tasks(), limit=2 * nproc),
                total=n_units, desc=f"prebuild mmap-frames x{nproc}", unit="vid",
            ):
                pass
        return len(pending), skipped

    def get_frames(
        self,
        *,
        trajectory_id: int,
        video_key: str,
        video_path: Path,
        frame_indices: np.ndarray,
        episode_timestamps: np.ndarray | None,
        from_timestamp: float,
        video_backend: str,
        video_backend_kwargs: dict | None,
    ) -> np.ndarray:
        arrays = self.get_frames_batch(
            [
                {
                    "trajectory_id": trajectory_id,
                    "video_key": video_key,
                    "video_path": video_path,
                    "frame_indices": frame_indices,
                    "episode_timestamps": episode_timestamps,
                    "from_timestamp": from_timestamp,
                    "video_backend": video_backend,
                    "video_backend_kwargs": video_backend_kwargs or {},
                }
            ]
        )
        return arrays[0]

    def gather_blobs_batch(self, items: list[dict]) -> tuple[list[memoryview], list[int], tuple[int, int]]:
        if not self.enabled:
            raise RuntimeError("MmapFrameStore is disabled")
        blobs: list[memoryview] = []
        counts: list[int] = []
        hw: tuple[int, int] | None = None
        for item in items:
            episode = self._load_or_build(
                trajectory_id=int(item["trajectory_id"]),
                video_key=str(item["video_key"]),
                video_path=Path(item["video_path"]),
                episode_timestamps=item.get("episode_timestamps"),
                from_timestamp=float(item.get("from_timestamp") or 0.0),
                video_backend=str(item.get("video_backend") or "decord"),
                video_backend_kwargs=item.get("video_backend_kwargs") or {},
            )
            if hw is None:
                hw = (episode.height, episode.width)
            elif hw != (episode.height, episode.width):
                raise RuntimeError(f"mixed JPEG sizes {(episode.height, episode.width)} vs {hw} in one decode batch")
            group = episode.gather_blobs(item["frame_indices"])
            counts.append(len(group))
            blobs.extend(group)
        if hw is None:
            hw = (0, 0)
        return blobs, counts, hw

    def decode_into(self, blobs: list[memoryview], out: np.ndarray) -> np.ndarray:
        return decode_jpegs_into(blobs, out, self._get_decode_pool())

    def get_frames_batch(self, items: list[dict]) -> list[np.ndarray]:
        blobs, counts, (height, width) = self.gather_blobs_batch(items)
        if not blobs:
            return [np.empty((0, height, width, 3), dtype=np.uint8) for _ in counts]
        packed = np.empty((len(blobs), height, width, 3), dtype=np.uint8)
        self.decode_into(blobs, packed)
        out: list[np.ndarray] = []
        cursor = 0
        for n in counts:
            out.append(packed[cursor : cursor + n])
            cursor += n
        return out

    def _load_or_build(
        self,
        *,
        trajectory_id: int,
        video_key: str,
        video_path: Path,
        episode_timestamps: np.ndarray | None,
        from_timestamp: float,
        video_backend: str,
        video_backend_kwargs: dict,
        progress: bool | None = None,
    ) -> MmapFrameEpisode:
        cache_dir = self.cache_dir_for(trajectory_id, video_key)
        source_tag = self._source_tag(video_path, episode_timestamps, from_timestamp)
        key = cache_dir.as_posix()
        if key in self._episodes:
            return self._episodes[key]

        if _cache_is_ready(cache_dir, source_tag):
            self._episodes[key] = MmapFrameEpisode.open(cache_dir)
            return self._episodes[key]

        if not self.allow_build:
            raise FileNotFoundError(
                f"mmap frame cache missing for episode {int(trajectory_id):06d} {video_key} "
                f"at {cache_dir}. Prebuild on rank 0 before training."
            )

        if progress is None:
            progress = _show_progress()
        with exclusive_cache_lock(cache_dir / ".lock"):
            if not _cache_is_ready(cache_dir, source_tag):
                self._build_cache(
                    cache_dir,
                    source_tag=source_tag,
                    video_path=video_path,
                    trajectory_id=trajectory_id,
                    video_key=video_key,
                    episode_timestamps=episode_timestamps,
                    from_timestamp=from_timestamp,
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs,
                    progress=progress,
                )
            self._episodes[key] = MmapFrameEpisode.open(cache_dir)
        return self._episodes[key]

    def _build_cache(
        self,
        cache_dir: Path,
        *,
        source_tag: str,
        video_path: Path,
        trajectory_id: int,
        video_key: str,
        episode_timestamps: np.ndarray | None,
        from_timestamp: float,
        video_backend: str,
        video_backend_kwargs: dict,
        progress: bool,
    ) -> None:
        pack = dict(
            cache_dir=cache_dir,
            source_tag=source_tag,
            jpeg_quality=self.jpeg_quality,
            image_size=self.image_size,
            desc=f"mmap-frames ep{int(trajectory_id):06d} {video_key}",
            progress=progress,
        )
        stills = _source_jpegs_of(self._decode_all_frames, video_backend_kwargs)
        if stills is not None:
            write_frame_cache(**pack, source_jpegs=stills)
            return
        frames, frame_iter = _rgb_for_cache(
            video_path,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            episode_timestamps=episode_timestamps,
            from_timestamp=from_timestamp,
            decode_all_frames=self._decode_all_frames,
            progress=progress,
        )
        write_frame_cache(**pack, frames=frames, frame_iter=frame_iter)

    def _source_tag(
        self,
        video_path: Path,
        episode_timestamps: np.ndarray | None,
        from_timestamp: float,
    ) -> str:
        stat = video_path.stat()
        n = 0 if episode_timestamps is None else len(episode_timestamps)
        size = "native" if self.image_size is None else str(int(self.image_size))
        return (
            f"{video_path.resolve()}#{stat.st_mtime_ns}:{stat.st_size}"
            f":n={n}:from={from_timestamp:.6f}:hw={size}:q={self.jpeg_quality}:jpeg"
        )
