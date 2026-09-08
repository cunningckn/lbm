"""JPEG-packed mmap video cache, matching lbm_origin shard IO.

Hot path: letterbox to ``image_size`` (224), JPEG quality 85, pack into
``frames.bin`` + offset/length tables, mmap + parallel cv2 decode into RGB.
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
    exclusive_cache_lock,
    read_manifest,
)

VIDEO_SUBDIR = "video"
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


_OPEN_STORES: weakref.WeakSet[MmapVideoStore] = weakref.WeakSet()


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


def _cache_is_ready(cache_dir: Path, source_tag: str) -> bool:
    if not all((cache_dir / name).is_file() for name in JPEG_FILES):
        return False
    manifest = read_manifest(cache_dir / MANIFEST)
    return (
        bool(manifest)
        and manifest.get("source") == source_tag
        and manifest.get("layout") == LAYOUT_JPEG
        and int(manifest.get("height") or 0) > 0
        and int(manifest.get("width") or 0) > 0
    )


def align_frames_to_parquet_steps(
    frames: np.ndarray,
    timestamps: np.ndarray | None,
    from_timestamp: float = 0.0,
) -> np.ndarray:
    """One cache slot per parquet step. Training indexes slots with ``step_indices``.

    LeRobot videos are the same fps as parquet. When the decoder returns the same
    number of frames as parquet rows, slots are 1:1. Extra or missing video frames
    are mapped with ``round(timestamp * fps)``.
    """
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


def write_video_cache(
    cache_dir: Path,
    *,
    source_tag: str,
    frames: np.ndarray,
    jpeg_quality: int = 85,
    image_size: int | None = None,
    desc: str | None = None,
    progress: bool | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    jpegs: list[bytes] = []
    if progress is None:
        progress = _show_progress()
    indices = range(len(frames))
    if progress:
        from lbm.utils.progress import track

        indices = track(indices, desc=desc or "mmap-video jpeg", total=len(frames), unit="f", leave=False)
    for i in indices:
        frame = frames[i]
        if image_size is not None:
            frame = resize_with_pad(frame, int(image_size))
        jpegs.append(encode_jpeg_rgb(frame, quality=jpeg_quality))
    blob, offset, length = pack_jpeg_frames(jpegs)
    atomic_write_bytes(cache_dir / "frames.bin", blob)
    atomic_save_npy(cache_dir / "offset.npy", offset)
    atomic_save_npy(cache_dir / "length.npy", length)
    sample = resize_with_pad(frames[0], int(image_size)) if image_size is not None else frames[0]
    manifest = {
        "source": source_tag,
        "layout": LAYOUT_JPEG,
        "num_frames": int(len(frames)),
        "jpeg_quality": int(jpeg_quality),
        "height": int(sample.shape[0]),
        "width": int(sample.shape[1]),
        "image_size": int(image_size) if image_size is not None else None,
    }
    atomic_write_text(cache_dir / MANIFEST, json.dumps(manifest, indent=2))


@dataclass
class MmapVideoEpisode:
    cache_dir: Path
    num_frames: int
    height: int
    width: int
    _blob: np.memmap
    _offset: np.ndarray
    _length: np.ndarray

    @classmethod
    def open(cls, cache_dir: Path) -> MmapVideoEpisode:
        manifest = read_manifest(cache_dir / MANIFEST)
        if manifest is None:
            raise FileNotFoundError(f"invalid video mmap manifest in {cache_dir}")
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


def _prebuild_video_job(job: dict) -> str:
    """Spawn-safe worker: decode one episode×camera into the JPEG mmap cache."""
    store = MmapVideoStore(
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
            progress=False,
        )
    finally:
        store.close()
    return f"{job['trajectory_id']}:{job['video_key']}"


class MmapVideoStore:
    """Build-once JPEG pack cache for episode video streams."""

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
        self.cache_root = self.dataset_path / MMAP_DIRNAME / VIDEO_SUBDIR
        self._episodes: dict[str, MmapVideoEpisode] = {}
        # Do not start threads here. DataLoader forks after Dataset.__init__;
        # a live ThreadPoolExecutor in the parent deadlocks the child.
        self._decode_pool: ThreadPoolExecutor | None = None
        self._decode_pool_pid: int | None = None
        self.allow_build = True
        _OPEN_STORES.add(self)

    def tune_decode_workers_for_loaders(self, n_loaders: int) -> None:
        """Cap per-worker JPEG threads so 16 loaders do not oversubscribe the machine."""
        if not self._decode_workers_auto:
            return
        n = decode_threads_for_loaders(n_loaders)
        if n == self.decode_workers:
            return
        self.decode_workers = n
        self._abandon_decode_pool()

    def _abandon_decode_pool(self) -> None:
        """Forget an inherited pool after fork. Do not shutdown: those threads are gone."""
        self._decode_pool = None
        self._decode_pool_pid = None

    def _get_decode_pool(self) -> ThreadPoolExecutor | None:
        """Process-local JPEG decode pool. Created after fork, never inherited."""
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
        tag = self._source_tag(
            Path(job["video_path"]),
            job.get("episode_timestamps"),
            float(job["from_timestamp"]),
        )
        return _cache_is_ready(cache_dir, tag)

    def prebuild(self, jobs: list[dict], *, workers: int = 0) -> tuple[int, int]:
        """Build missing JPEG packs. Rank-0 only; training should then set ``allow_build=False``."""
        pending = [job for job in jobs if not self.job_ready(job)]
        skipped = len(jobs) - len(pending)
        if not pending:
            from lbm.utils.progress import progress_enabled

            if skipped and progress_enabled():
                print(f"mmap-video: {skipped} already cached", flush=True)
            return 0, skipped
        nproc = workers if workers and workers > 0 else default_prebuild_workers()
        nproc = max(1, min(int(nproc), len(pending)))
        from lbm.utils.progress import track

        if nproc == 1:
            for job in track(pending, desc="prebuild mmap-video", unit="vid"):
                _prebuild_video_job(job)
            return len(pending), skipped
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as pool:
            futures = [pool.submit(_prebuild_video_job, job) for job in pending]
            for fut in track(
                as_completed(futures), total=len(futures), desc=f"prebuild mmap-video x{nproc}", unit="vid"
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
        """Collect JPEG memoryviews for many (camera, indices) groups. No decode."""
        if not self.enabled:
            raise RuntimeError("MmapVideoStore is disabled")
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
        """Decode JPEG blobs as RGB into a preallocated ``(N, H, W, 3)`` array."""
        return decode_jpegs_into(blobs, out, self._get_decode_pool())

    def get_frames_batch(self, items: list[dict]) -> list[np.ndarray]:
        """Decode many (camera, indices) JPEG groups in one thread-pool pass."""
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
    ) -> MmapVideoEpisode:
        cache_dir = self.cache_dir_for(trajectory_id, video_key)
        source_tag = self._source_tag(video_path, episode_timestamps, from_timestamp)
        key = cache_dir.as_posix()
        if key in self._episodes:
            return self._episodes[key]

        if _cache_is_ready(cache_dir, source_tag):
            self._episodes[key] = MmapVideoEpisode.open(cache_dir)
            return self._episodes[key]

        if not self.allow_build:
            raise FileNotFoundError(
                f"mmap video cache missing for episode {int(trajectory_id):06d} {video_key} "
                f"at {cache_dir}. Prebuild on rank 0 before training."
            )

        if progress is None:
            progress = _show_progress()
        with exclusive_cache_lock(cache_dir / ".lock"):
            if not _cache_is_ready(cache_dir, source_tag):
                decode = self._decode_all_frames or get_all_frames
                frames = decode(
                    video_path.as_posix(),
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs,
                    progress=progress,
                )
                frames = align_frames_to_parquet_steps(frames, episode_timestamps, from_timestamp)
                write_video_cache(
                    cache_dir,
                    source_tag=source_tag,
                    frames=frames,
                    jpeg_quality=self.jpeg_quality,
                    image_size=self.image_size,
                    desc=f"mmap-video ep{int(trajectory_id):06d} {video_key}",
                    progress=progress,
                )
            self._episodes[key] = MmapVideoEpisode.open(cache_dir)
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
