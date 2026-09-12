"""In-memory custom episodes → packed training samples."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from lbm.action_space import (
    DEFAULT,
    EEF,
    JOINT,
    XYZ_ROTVEC,
    actions_in_train_space,
    native_format,
    pack_to_format,
    parse_format,
    parse_kind,
    parse_rep,
    resolve_action_space,
    stores_file_delta,
)
from lbm.dataloader.custom.common.lerobot import lerobot_of, read_lerobot_frames, read_lerobot_vectors
from lbm.dataloader.custom.spec import CustomSpec
from lbm.temporal import delta_indices, n_steps, native_stride

_log = logging.getLogger(__name__)
_FILE_DELTA_LOGGED: set[str] = set()


def _lock_file_delta_action_freq(spec: CustomSpec, action_mode: str, freq: float, *, requested: float | None) -> float:
    slices = resolve_action_space(spec, action_mode)
    if not stores_file_delta(slices):
        return freq
    locked = float(spec.fps)
    stride = native_stride(locked, freq)
    overridden = stride != 1 or (requested is not None and abs(float(requested) - locked) > 1e-6)
    if overridden:
        msg = (
            f"[data] {spec.name}: on-disk action is per-frame delta; "
            f"action_freq locked to {locked:g} Hz (stride=1), ignored {freq:g} Hz"
        )
        _log.warning(msg)
        print(msg, flush=True)
        return locked
    if spec.name not in _FILE_DELTA_LOGGED:
        _FILE_DELTA_LOGGED.add(spec.name)
        msg = f"[data] {spec.name}: on-disk action is per-frame delta; action_freq pinned to {locked:g} Hz"
        _log.info(msg)
        print(msg, flush=True)
    return locked


@dataclass
class Episode:
    images: dict[str, np.ndarray]
    state: np.ndarray
    action: np.ndarray
    lang: str = ""


def _resize_u8(frames: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.shape[1] == size and arr.shape[2] == size:
        return arr.astype(np.uint8, copy=False)
    try:
        import cv2
    except ImportError:
        return _nearest_resize(arr, size)
    out = np.empty((arr.shape[0], size, size, arr.shape[-1]), dtype=np.uint8)
    for i, frame in enumerate(arr):
        out[i] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return out


def _nearest_resize(frames: np.ndarray, size: int) -> np.ndarray:
    t, h, w, c = frames.shape
    ys = (np.linspace(0, h - 1, size)).astype(np.int64)
    xs = (np.linspace(0, w - 1, size)).astype(np.int64)
    return np.ascontiguousarray(frames[:, ys][:, :, xs].astype(np.uint8))


class CustomSingleDataset(Dataset):
    """Window episodes (in-memory or lazy on-disk records) into packed samples."""

    def __init__(
        self,
        spec: CustomSpec,
        episodes: list[Episode] | None = None,
        *,
        records: list | None = None,
        action_length: float = 5.0,
        action_freq: float | None = None,
        history_length: float = 0.0,
        history_freq: float | None = None,
        image_size: int | None = None,
        use_mmap: bool = False,
        use_mmap_frames: bool | None = None,
        mmap_jpeg_quality: int = 85,
        root: Path | str | None = None,
        norm_stats: dict[str, Any] | None = None,
        action_mode: str,
        action_kind: str | None = None,
        action_format: str | None = None,
    ) -> None:
        if not episodes and not records:
            raise ValueError(f"{spec.name}: no episodes")
        self.spec = spec
        self.root = Path(root) if root is not None else None
        self.episodes = episodes or []
        self.records = records or []
        self.action_mode = parse_rep(action_mode)
        kind = parse_kind(action_kind) if action_kind else None
        self.action_kind = None if kind in (None, JOINT) else kind
        self.action_length = float(action_length)
        freq = float(action_freq or spec.fps)
        freq = _lock_file_delta_action_freq(spec, self.action_mode, freq, requested=action_freq)
        self.action_freq = freq
        if spec.fps > 0 and freq > 0:
            ratio = float(spec.fps) / float(freq)
            if abs(ratio - round(ratio)) > 0.05:
                msg = (
                    f"[data] {spec.name}: native fps {spec.fps:g} / action_freq {freq:g} "
                    f"= {ratio:.4g} is not near an integer; stride={native_stride(spec.fps, freq)}"
                )
                _log.warning(msg)
                print(msg, flush=True)
        if self.action_kind == EEF:
            fmt = parse_format(action_format) if action_format else XYZ_ROTVEC
            self.action_format = XYZ_ROTVEC if fmt == DEFAULT else fmt
            from lbm.kinematics import apply_joint_fk

            z_st, z_act, canon = apply_joint_fk(
                np.zeros((1, spec.state_dim), np.float32),
                np.zeros((1, spec.action_dim), np.float32),
                spec,
                self.action_mode,
                self.action_kind,
                action_format=XYZ_ROTVEC,
            )
            z_st, z_act, slices = pack_to_format(z_st, z_act, canon, self.action_format)
        else:
            self.action_format = (
                parse_format(action_format)
                if action_format
                else native_format(tuple(spec.action_space))
            )
            native = resolve_action_space(spec, self.action_mode)
            z_st, z_act, slices = pack_to_format(
                np.zeros((1, spec.state_dim), np.float32),
                np.zeros((1, spec.action_dim), np.float32),
                native,
                self.action_format,
            )
        self._space = slices
        self._io_state_dim = int(z_st.shape[-1])
        self._io_action_dim = int(z_act.shape[-1])
        if norm_stats is None and self.root is not None:
            from lbm.utils.preprocess import load_dump_norm_stats

            norm_stats = load_dump_norm_stats(self.root, freq, self.action_length, slices)
        self.norm_stats = norm_stats
        hist_freq = float(history_freq or spec.fps)
        self.chunk_length = n_steps(self.action_length, freq)
        self.action_deltas = delta_indices(self.action_length, freq, spec.fps, past=False)
        self.history_deltas = delta_indices(history_length, hist_freq, spec.fps, past=True)
        self.image_size = int(image_size or spec.image_size)
        self._use_mmap = bool(use_mmap)
        self._use_mmap_frames = bool(use_mmap if use_mmap_frames is None else use_mmap_frames)
        self._mmap_jpeg_quality = int(mmap_jpeg_quality)
        self._mmap_allow_build = True
        self._mmap_stores: dict[str, object] = {}
        self._parquet_stores: dict[str, object] = {}
        self._index: list[tuple[int, int]] | None = None
        self._offsets: np.ndarray | None = None
        self._vec_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if self.episodes:
            self._offsets = np.cumsum([len(e.action) for e in self.episodes], dtype=np.int64)
        else:
            lengths = np.array([max(int(r.n_frames), 0) for r in self.records], dtype=np.int64)
            if not len(lengths) or int(lengths.sum()) <= 0:
                raise ValueError(f"{spec.name}: no frames")
            self._offsets = np.cumsum(lengths)

    def __len__(self) -> int:
        if self._index is not None:
            return len(self._index)
        return int(self._offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if self._index is not None:
            return self._index[int(index)]
        index = int(index)
        epi = int(np.searchsorted(self._offsets, index, side="right"))
        prev = 0 if epi == 0 else int(self._offsets[epi - 1])
        return epi, index - prev

    @property
    def policy_io(self) -> dict[str, Any]:
        io = self.spec.as_policy_io(chunk_length=self.chunk_length)
        io["action_dim"] = int(self._io_action_dim)
        io["state_dim"] = int(self._io_state_dim)
        return io

    def _parquet_store(self, repo: Path | str):
        key = str(Path(repo).resolve())
        store = self._parquet_stores.get(key)
        if store is None:
            from lbm.dataloader.mmap.mmap_io import MmapTrajectoryStore

            store = MmapTrajectoryStore(repo, enabled=self._use_mmap)
            self._parquet_stores[key] = store
        return store

    def _vector_kwargs(self) -> dict[str, Any]:
        return {"action_freq": self.action_freq}

    def _read_vectors(self, record) -> tuple[np.ndarray, np.ndarray]:
        dump = lerobot_of(record)
        kwargs = self._vector_kwargs()
        if self._use_mmap and dump is not None:
            return read_lerobot_vectors(record, self.spec, mmap_store=self._parquet_store(dump.repo), **kwargs)
        from lbm.dataloader.custom.scan import read_vectors

        return read_vectors(record, self.spec, **kwargs)

    def _vectors(self, epi_i: int) -> tuple[np.ndarray, np.ndarray]:
        if self.episodes:
            ep = self.episodes[epi_i]
            return np.asarray(ep.state), np.asarray(ep.action)
        if epi_i not in self._vec_cache:
            self._vec_cache[epi_i] = self._read_vectors(self.records[epi_i])
            if len(self._vec_cache) > 8:
                self._vec_cache.pop(next(iter(self._vec_cache)))
        return self._vec_cache[epi_i]

    def _fk_cached(self, epi_i: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.root is None or not self.records:
            return None
        from lbm.dataloader.custom.fk_cache import load_fk_episode, source_key

        rec = self.records[epi_i]
        return load_fk_episode(self.root, epi_i, source=source_key(rec), n_frames=rec.n_frames)

    def _policy_vectors(self, epi_i: int) -> tuple[np.ndarray, np.ndarray, tuple]:
        from lbm.dataloader.custom.fk_cache import needs_joint_fk

        if self.action_kind == EEF and needs_joint_fk(self.spec):
            cached = self._fk_cached(epi_i)
            if cached is not None:
                canon = resolve_action_space(
                    self.spec, self.action_mode, action_kind=self.action_kind, action_format=XYZ_ROTVEC
                )
                return pack_to_format(cached[0], cached[1], canon, self.action_format)
            state, action = self._vectors(epi_i)
            from lbm.kinematics import apply_joint_fk

            state, action, canon = apply_joint_fk(
                state, action, self.spec, self.action_mode, self.action_kind, action_format=XYZ_ROTVEC
            )
            return pack_to_format(state, action, canon, self.action_format)
        state, action = self._vectors(epi_i)
        native = resolve_action_space(self.spec, self.action_mode)
        return pack_to_format(state, action, native, self.action_format)

    def __getitem__(self, index: int) -> dict[str, Any]:
        epi_i, t = self._locate(index)
        state_arr, action_arr, slices = self._policy_vectors(epi_i)
        n = int(action_arr.shape[0])
        action = self._gather_vector(action_arr, t, self.action_deltas, n)
        state = state_arr[min(t, n - 1)]
        images = []
        camera_mask = []
        for cam in self.spec.camera_keys:
            gathered, present = self._cam_frames(epi_i, cam, t, n)
            images.append(_resize_u8(gathered, self.image_size))
            camera_mask.append(present)
        lang = self.episodes[epi_i].lang if self.episodes else self.records[epi_i].lang
        action = action.astype(np.float32, copy=False)
        state = np.asarray(state, dtype=np.float32)
        action = actions_in_train_space(action, state, slices)
        if self.norm_stats is not None:
            from lbm.utils.preprocess import normalize

            action = np.asarray(normalize(action, self.norm_stats["actions"], slices), dtype=np.float32)
            state = np.asarray(
                normalize(state, self.norm_stats["state"], slices, field="state"), dtype=np.float32
            )
        return {
            "image": images,
            "action": action,
            "state": state,
            "lang": lang,
            "camera_keys": self.spec.camera_keys,
            "camera_mask": np.asarray(camera_mask, dtype=bool),
            "robot_tag": self.spec.embodiment,
            "embodiment_id": int(self.spec.embodiment_id),
        }

    def _gather_vector(self, arr: np.ndarray, t: int, deltas: np.ndarray, n: int) -> np.ndarray:
        idx = np.clip(t + deltas.astype(np.int64), 0, n - 1)
        return np.asarray(arr)[idx]

    def _gather_frames(self, frames: np.ndarray, t: int, n: int) -> np.ndarray:
        arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None]
        idx = np.clip(t + self.history_deltas.astype(np.int64), 0, min(n, arr.shape[0]) - 1)
        return arr[idx]

    def _blank_history(self) -> np.ndarray:
        n_hist = int(self.history_deltas.shape[0])
        size = self.image_size
        return np.zeros((n_hist, size, size, 3), dtype=np.uint8)

    def _cam_frames(self, epi_i: int, cam: str, t: int, n: int) -> tuple[np.ndarray, bool]:
        """History window for ``cam``. Missing views are black zeros, not copies of another cam."""
        if self.episodes:
            frames = self.episodes[epi_i].images.get(cam)
            if frames is None:
                return self._blank_history(), False
            return self._gather_frames(frames, t, n), True
        record = self.records[epi_i]
        from lbm.dataloader.custom.mmap_frames import camera_has_source

        if not camera_has_source(record, self.spec, cam):
            return self._blank_history(), False
        idx = np.clip(t + self.history_deltas.astype(np.int64), 0, max(n - 1, 0)).tolist()
        try:
            gathered = self._read_record_frames(record, cam, idx)
        except KeyError:
            return self._blank_history(), False
        return gathered, True

    def _mmap_store(self, cache_root: Path):
        if not self._use_mmap_frames:
            return None
        key = str(cache_root)
        store = self._mmap_stores.get(key)
        if store is None:
            from lbm.dataloader.custom.scan import decode_custom_mmap
            from lbm.dataloader.mmap.frame_mmap_io import MmapFrameStore

            store = MmapFrameStore(
                cache_root,
                enabled=True,
                jpeg_quality=self._mmap_jpeg_quality,
                image_size=self.image_size,
                decode_all_frames=decode_custom_mmap,
            )
            store.allow_build = self._mmap_allow_build
            self._mmap_stores[key] = store
        return store

    def _read_live_frames(self, record, cam: str, indices: list[int]) -> np.ndarray:
        dump = lerobot_of(record)
        if dump is not None:
            mmap_store = self._parquet_store(dump.repo) if self._use_mmap else None
            return read_lerobot_frames(record, self.spec, cam, indices, mmap_store=mmap_store)
        from lbm.dataloader.custom.scan import read_frames

        return read_frames(record, self.spec, cam, indices)

    def _job_from_ref(self, ref):
        from lbm.dataloader.custom.mmap_frames import decode_kwargs

        store = self._mmap_store(ref.cache_root)
        if store is None or not store.enabled:
            return ref, None, None
        job = store.make_job(
            trajectory_id=int(ref.trajectory_id),
            video_key=str(ref.video_key),
            video_path=ref.path,
            episode_timestamps=None,
            from_timestamp=0.0,
            video_backend="custom",
            video_backend_kwargs=decode_kwargs(ref, self.spec),
        )
        return ref, store, job

    def _frame_mmap_job(self, record, cam: str):
        from lbm.dataloader.custom.mmap_frames import resolve_mmap_frames

        ref = resolve_mmap_frames(record, self.spec, cam, dump_root=self.root)
        if ref is None:
            return None, None, None
        return self._job_from_ref(ref)

    def _read_record_frames(self, record, cam: str, indices: list[int]) -> np.ndarray:
        ref, store, job = self._frame_mmap_job(record, cam)
        if store is not None and job is not None and ref is not None:
            if store.job_ready(job) or (store.allow_build and ref.path.exists()):
                return store.get_frames(
                    trajectory_id=int(job["trajectory_id"]),
                    video_key=str(job["video_key"]),
                    video_path=Path(job["video_path"]),
                    frame_indices=np.asarray([int(i) for i in indices], dtype=np.int64),
                    episode_timestamps=job.get("episode_timestamps"),
                    from_timestamp=float(job["from_timestamp"]),
                    video_backend=str(job["video_backend"]),
                    video_backend_kwargs=job.get("video_backend_kwargs") or {},
                )
        return self._read_live_frames(record, cam, indices)

    def prebuild_mmap_caches(self, *, workers: int = 0) -> None:
        self._prebuild_parquet_mmap(workers=workers)
        self._prebuild_frame_mmap(workers=workers)

    def prebuild_fk_cache(self, *, workers: int = 0, force: bool = False) -> None:
        from lbm.dataloader.custom.fk_cache import prebuild_fk

        prebuild_fk(self, workers=workers, force=force)

    def _prebuild_parquet_mmap(self, *, workers: int = 0) -> None:
        if not self._use_mmap or not self.records:
            return
        v2: dict[str, list] = {}
        v3: dict[str, list] = {}
        for record in self.records:
            dump = lerobot_of(record)
            if dump is None:
                continue
            repo = str(dump.repo)
            if dump.is_v3:
                v3.setdefault(repo, []).append((dump.parquet, dump.episode_index))
            else:
                v2.setdefault(repo, []).append(dump.parquet)
        built = skipped = 0
        for repo, paths in v2.items():
            b, s = self._parquet_store(repo).prebuild_files(paths, workers=workers)
            built += b
            skipped += s
        for repo, items in v3.items():
            b, s = self._parquet_store(repo).prebuild_episode_rows(items, workers=workers)
            built += b
            skipped += s
        print(f"[mmap] {self.spec.name}: parquet built={built} reused={skipped}")

    def _prebuild_frame_mmap(self, *, workers: int = 0) -> None:
        from lbm.dataloader.custom.mmap_frames import collect_frame_refs

        grouped: dict[str, tuple[Any, list]] = {}
        for ref in collect_frame_refs(self.records, self.spec, dump_root=self.root):
            _ref, store, job = self._job_from_ref(ref)
            if store is None or job is None:
                continue
            grouped.setdefault(str(store.dataset_path), (store, []))[1].append(job)
        n_jobs = sum(len(jobs) for _, jobs in grouped.values())
        if not n_jobs:
            print(f"[mmap] {self.spec.name}: no mp4/parquet-image streams to cache")
            return
        from lbm.dataloader.mmap.frame_mmap_io import split_pending_frame_jobs

        pending, skipped = split_pending_frame_jobs(
            grouped, desc=f"mmap-frames ready {self.spec.name}"
        )
        built = 0
        for store, jobs in pending.values():
            if not jobs:
                continue
            b, _s = store.prebuild(jobs, workers=workers, check_ready=False)
            built += int(b)
        print(f"[mmap] {self.spec.name}: frames built={built} reused={skipped} jobs={n_jobs}")

    def set_mmap_allow_build(self, allow: bool) -> None:
        self._mmap_allow_build = bool(allow)
        for store in self._mmap_stores.values():
            store.allow_build = bool(allow)


class CustomMixtureDataset(Dataset):
    """Flattened mixture of custom datasets (heterogeneous IO is padded in collate)."""

    def __init__(
        self,
        pairs: list[tuple[CustomSingleDataset, float]],
        *,
        mode: str = "train",
        seed: int = 0,
        **_kwargs: Any,
    ) -> None:
        if not pairs:
            raise ValueError("mixture is empty")
        self.datasets = [ds for ds, _ in pairs]
        self.weights = [float(w) for _, w in pairs]
        self.mode = mode
        self.seed = int(seed)
        if any(not np.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("mixture weights must be positive and finite")
        self._offsets = np.cumsum([len(ds) for ds in self.datasets], dtype=np.int64)
        if not self._offsets[-1] or any(len(ds) == 0 for ds in self.datasets):
            raise ValueError("mixture has no samples or contains an empty source")

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def spec(self) -> CustomSpec:
        return self.datasets[0].spec

    @property
    def chunk_length(self) -> int:
        return max(ds.chunk_length for ds in self.datasets)

    @property
    def policy_io(self) -> dict[str, Any]:
        from lbm.batch import merge_policy_io

        return merge_policy_io([ds.policy_io for ds in self.datasets])

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index) % len(self)
        ds_i = int(np.searchsorted(self._offsets, index, side="right"))
        local = index - (int(self._offsets[ds_i - 1]) if ds_i else 0)
        return self.datasets[ds_i][local]

    def prebuild_mmap_caches(self, workers: int = 0) -> None:
        from lbm.utils.progress import track

        datasets = self.datasets
        if len(datasets) > 1:
            datasets = track(datasets, desc="mmap dumps", unit="dump")
        for dataset in datasets:
            dataset.prebuild_mmap_caches(workers=workers)

    def prebuild_fk_caches(self, *, workers: int = 0, force: bool = False) -> None:
        from lbm.utils.progress import track

        datasets = self.datasets
        if len(datasets) > 1:
            datasets = track(datasets, desc="fk dumps", unit="dump")
        for dataset in datasets:
            dataset.prebuild_fk_cache(workers=workers, force=force)

    def set_mmap_allow_build(self, allow: bool) -> None:
        for dataset in self.datasets:
            dataset.set_mmap_allow_build(allow)
