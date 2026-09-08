"""JPEG-packed mmap frame cache.

Hot path: letterbox to ``image_size`` (224) at pack time, JPEG quality 85,
``frames.bin`` + offset/length tables, mmap + parallel cv2 decode into RGB.

Decode is always native resolution. ``write_frame_cache`` pads to square.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    atomic_write_bytes,
    atomic_write_text,
    cache_files_present,
    exclusive_cache_lock,
    read_manifest,
    read_source_manifest,
)

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
    path = Path(video_path)
    if not path.is_file():
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    frames = _decode_av_all(path, progress=progress)
    if frames is None or frames.shape[0] == 0:
        frames = _decode_cv2_all(path, progress=progress)
    if frames is None or frames.shape[0] == 0:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    if resize_size is not None:
        h, w = int(resize_size[0]), int(resize_size[1])
        frames = _resize_thwc(frames, h, w)
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


def _decode_av_all(path: Path, *, progress: bool = False) -> np.ndarray | None:
    try:
        import av
    except ImportError:
        return None
    try:
        av.logging.set_level(av.logging.ERROR)
    except Exception:
        pass
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
    except Exception:
        return None
    iterator = container.decode(stream)
    n = int(stream.frames or 0) or None
    if progress:
        from lbm.utils.progress import track

        iterator = track(iterator, desc=f"decode {path.name}", total=n, unit="f", leave=False)
    frames = [frame.to_ndarray(format="rgb24") for frame in iterator]
    container.close()
    return np.stack(frames, axis=0) if frames else None


def _decode_cv2_all(path: Path, *, progress: bool = False) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    out: list[np.ndarray] = []
    bar = None
    if progress:
        from lbm.utils.progress import progress_bar

        bar = progress_bar(total=n, desc=f"decode {path.name}", unit="f", leave=False)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
        cap.release()
    return np.stack(out, axis=0) if out else None


def _decode_from_job(job: dict):
    mod = job.get("decode_module")
    name = job.get("decode_name")
    if not mod or not name:
        return None
    import importlib

    return getattr(importlib.import_module(str(mod)), str(name))


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


def _jpeg_manifest_ready(cache_dir: Path, source_tag: str) -> bool:
    manifest = read_source_manifest(cache_dir, source_tag)
    return (
        bool(manifest)
        and manifest.get("layout") == LAYOUT_JPEG
        and int(manifest.get("height") or 0) > 0
        and int(manifest.get("width") or 0) > 0
    )


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
    count = len(jpegs)
    offset = np.zeros(count, dtype=np.int64)
    length = np.zeros(count, dtype=np.uint32)
    chunks: list[bytes] = []
    cursor = 0
    for i, data in enumerate(jpegs):
        offset[i] = cursor
        length[i] = len(data)
        chunks.append(data)
        cursor += len(data)
    return b"".join(chunks), offset, length


def write_frame_cache(
    cache_dir: Path,
    *,
    source_tag: str,
    frames: np.ndarray | None = None,
    source_jpegs: list[bytes] | None = None,
    jpeg_quality: int = 85,
    image_size: int | None = None,
    desc: str | None = None,
    progress: bool | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if source_jpegs is not None:
        items: list | np.ndarray = source_jpegs
        stills = True
    elif frames is not None:
        items = np.ascontiguousarray(frames, dtype=np.uint8)
        stills = False
    else:
        raise ValueError("write_frame_cache needs frames or source_jpegs")
    n = len(items)
    if n <= 0:
        raise ValueError("write_frame_cache got no frames")
    jpegs: list[bytes] = []
    sample: np.ndarray | None = None
    if progress is None:
        progress = _show_progress()
    indices = range(n)
    if progress:
        from lbm.utils.progress import track

        indices = track(indices, desc=desc or "mmap-frames jpeg", total=n, unit="f", leave=False)
    for i in indices:
        frame = decode_still_rgb(items[i]) if stills else items[i]
        if image_size is not None:
            frame = resize_with_pad(frame, int(image_size))
        if sample is None:
            sample = frame
        jpegs.append(encode_jpeg_rgb(frame, quality=jpeg_quality))
    blob, offset, length = pack_jpeg_frames(jpegs)
    atomic_write_bytes(cache_dir / "frames.bin", blob)
    atomic_save_npy(cache_dir / "offset.npy", offset)
    atomic_save_npy(cache_dir / "length.npy", length)
    assert sample is not None
    manifest = {
        "source": source_tag,
        "layout": LAYOUT_JPEG,
        "num_frames": int(n),
        "jpeg_quality": int(jpeg_quality),
        "height": int(sample.shape[0]),
        "width": int(sample.shape[1]),
        "image_size": int(image_size) if image_size is not None else None,
    }
    atomic_write_text(cache_dir / MANIFEST, json.dumps(manifest, indent=2))


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
        blob = np.memmap(cache_dir / "frames.bin", dtype=np.uint8, mode="r")
        offset = np.load(cache_dir / "offset.npy", mmap_mode="r")
        length = np.load(cache_dir / "length.npy", mmap_mode="r")
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
        mmap = getattr(self._blob, "_mmap", None)
        if mmap is not None:
            mmap.close()

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


def lerobot_v3_file_span(job: dict) -> tuple[int, int] | None:
    """Packed-file ``[start, stop)`` from v3 ``from_timestamp``. None if not shareable."""
    kw = job.get("video_backend_kwargs") or {}
    if str(kw.get("mode") or "") != "lerobot":
        return None
    dump = (kw.get("extra") or {}).get("lerobot")
    cam = kw.get("cam")
    n = int(kw.get("n_frames") or 0)
    if dump is None or cam is None or n <= 0 or not getattr(dump, "is_v3", False):
        return None
    try:
        idxs = dump.mp4_indices(str(cam), list(range(n)))
    except Exception:
        return None
    if not idxs:
        return None
    start = int(idxs[0])
    if start < 0 or any(int(v) != start + i for i, v in enumerate(idxs)):
        return None
    return start, start + len(idxs)


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


def _source_jpegs_of(decode_fn, kwargs: dict) -> list[bytes] | None:
    if decode_fn is None:
        return None
    import importlib

    getter = getattr(importlib.import_module(decode_fn.__module__), "source_jpegs_for_mmap", None)
    if getter is None:
        return None
    return getter(kwargs)


def _write_job_rgb(store: MmapFrameStore, job: dict, frames: np.ndarray, *, progress: bool = False) -> None:
    timestamps = job.get("episode_timestamps")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
    frames = align_frames_to_parquet_steps(frames, timestamps, float(job["from_timestamp"]))
    cache_dir = store.cache_dir_for(int(job["trajectory_id"]), str(job["video_key"]))
    source_tag = store._source_tag(
        Path(job["video_path"]),
        job.get("episode_timestamps"),
        float(job["from_timestamp"]),
    )
    with exclusive_cache_lock(cache_dir / ".lock"):
        if _cache_is_ready(cache_dir, source_tag):
            return
        write_frame_cache(
            cache_dir,
            source_tag=source_tag,
            frames=frames,
            jpeg_quality=store.jpeg_quality,
            image_size=store.image_size,
            desc=f"mmap-frames ep{int(job['trajectory_id']):06d} {job['video_key']}",
            progress=progress,
        )


def write_shared_mp4_caches(store: MmapFrameStore, jobs: list[dict], *, progress: bool = False) -> None:
    """Decode one packed mp4 once and write each episode's JPEG cache."""
    from lbm.dataloader.custom.video import iter_mp4_span

    items: list[tuple[dict, tuple[int, int]]] = []
    for job in jobs:
        span = lerobot_v3_file_span(job)
        if span is None:
            _prebuild_frame_job({**job, "progress": progress})
            continue
        items.append((job, span))
    if not items:
        return
    items.sort(key=lambda item: item[1][0])
    min_start = items[0][1][0]
    max_stop = max(span[1] for _job, span in items)
    path = Path(items[0][0]["video_path"])
    pending = [(job, start, stop, []) for job, (start, stop) in items]
    file_i = min_start
    for frame in iter_mp4_span(path, min_start, max_stop, progress=progress):
        keep = []
        for job, start, stop, buf in pending:
            if start <= file_i < stop:
                buf.append(frame)
                if len(buf) == stop - start:
                    _write_job_rgb(store, job, np.stack(buf, axis=0), progress=False)
                    continue
            keep.append((job, start, stop, buf))
        pending = keep
        file_i += 1
        if not pending:
            break
    for job, _start, _stop, _buf in pending:
        _prebuild_frame_job({**job, "progress": progress})


def _shared_payload(store: MmapFrameStore, jobs: list[dict], *, progress: bool = False) -> dict:
    return {
        "dataset_path": str(store.dataset_path),
        "jpeg_quality": int(store.jpeg_quality),
        "image_size": store.image_size,
        "jobs": jobs,
        "progress": progress,
        "video_path": jobs[0]["video_path"],
    }


def _prebuild_shared_file_job(payload: dict) -> str:
    jobs = payload["jobs"]
    store = MmapFrameStore(
        payload["dataset_path"],
        jpeg_quality=int(payload["jpeg_quality"]),
        image_size=payload["image_size"],
        decode_workers=1,
        decode_all_frames=_decode_from_job(jobs[0]),
    )
    try:
        write_shared_mp4_caches(store, jobs, progress=bool(payload.get("progress")))
    finally:
        store.close()
    return str(payload["video_path"])


def _prebuild_frame_job(job: dict) -> str:
    """Spawn-safe worker: decode one episode×camera into the JPEG mmap cache."""
    store = MmapFrameStore(
        job["dataset_path"],
        jpeg_quality=int(job["jpeg_quality"]),
        image_size=job["image_size"],
        decode_workers=1,
        decode_all_frames=_decode_from_job(job),
    )
    timestamps = job.get("episode_timestamps")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
    try:
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
    finally:
        store.close()
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
        from lbm.utils.progress import track

        if nproc == 1:
            for group in shared:
                write_shared_mp4_caches(self, group, progress=True)
            for job in track(solo, desc="prebuild mmap-frames", unit="vid") if solo else ():
                _prebuild_frame_job({**job, "progress": True})
            return len(pending), skipped
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as pool:
            futures = [pool.submit(_prebuild_shared_file_job, _shared_payload(self, group)) for group in shared]
            futures += [pool.submit(_prebuild_frame_job, job) for job in solo]
            for fut in track(
                as_completed(futures), total=len(futures), desc=f"prebuild mmap-frames x{nproc}", unit="vid"
            ):
                fut.result()
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
                jpegs = _source_jpegs_of(self._decode_all_frames, video_backend_kwargs)
                if jpegs is not None:
                    write_frame_cache(
                        cache_dir,
                        source_tag=source_tag,
                        source_jpegs=jpegs,
                        jpeg_quality=self.jpeg_quality,
                        image_size=self.image_size,
                        desc=f"mmap-frames ep{int(trajectory_id):06d} {video_key}",
                        progress=progress,
                    )
                else:
                    decode = self._decode_all_frames or get_all_frames
                    frames = decode(
                        video_path.as_posix(),
                        video_backend=video_backend,
                        video_backend_kwargs=video_backend_kwargs,
                        progress=progress,
                    )
                    frames = align_frames_to_parquet_steps(frames, episode_timestamps, from_timestamp)
                    write_frame_cache(
                        cache_dir,
                        source_tag=source_tag,
                        frames=frames,
                        jpeg_quality=self.jpeg_quality,
                        image_size=self.image_size,
                        desc=f"mmap-frames ep{int(trajectory_id):06d} {video_key}",
                        progress=progress,
                    )
            self._episodes[key] = MmapFrameEpisode.open(cache_dir)
        return self._episodes[key]

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
