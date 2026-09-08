"""Memory-mapped LeRobot parquet cache for fast proprio / timestamp IO."""

from __future__ import annotations

import fcntl
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MMAP_DIRNAME = ".mmap"
MANIFEST = "manifest.json"


@contextmanager
def exclusive_cache_lock(lock_path: Path) -> Iterator[None]:
    """Process-level exclusive lock so DataLoader workers do not race cache builds."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Write ``.npy`` atomically so a crash cannot leave a half-written table."""
    import io

    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


def read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_source_manifest(cache_dir: Path, source: str) -> dict[str, Any] | None:
    """Manifest if it exists and ``source`` matches; else None."""
    manifest = read_manifest(cache_dir / MANIFEST)
    if manifest is None or manifest.get("source") != source:
        return None
    return manifest


def cache_files_present(folder: Path, names: list[str] | tuple[str, ...]) -> bool:
    try:
        return all((folder / name).is_file() for name in names)
    except OSError:
        return False


class _IlocIndexer:
    def __init__(self, column: _ColumnView):
        self._column = column

    def __getitem__(self, index: int) -> Any:
        return self._column._value_at(int(index))


class _ColumnView:
    """Minimal pandas-like column accessor for mmap-backed trajectory data."""

    def __init__(self, values: np.ndarray, *, stacked: bool = False):
        self._values = values
        self._stacked = stacked

    @property
    def iloc(self) -> _IlocIndexer:
        return _IlocIndexer(self)

    def to_numpy(self) -> np.ndarray:
        return self._values

    def tolist(self) -> list[Any]:
        if self._stacked:
            return list(self._values)
        return self._values.tolist()

    def _value_at(self, index: int) -> Any:
        value = self._values[int(index)]
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return value
        return value

    def __iter__(self):
        if self._stacked:
            yield from self._values
        else:
            yield from self._values.tolist()


@dataclass
class TrajectoryDataView:
    """DataFrame-compatible view over mmap-backed episode columns."""

    _columns: dict[str, _ColumnView]

    @property
    def columns(self):
        return self._columns.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._columns

    def __getitem__(self, key: str) -> _ColumnView:
        return self._columns[key]


def _is_stackable_object_series(series: pd.Series) -> bool:
    if series.dtype != object or series.empty:
        return False
    sample = series.iloc[0]
    return isinstance(sample, (list, tuple, np.ndarray))


def _stack_object_series(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=np.float32) for x in series], axis=0)


def _mmap_array_for_series(series: pd.Series) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Numeric / stacked vector columns only. PNG/JPEG bytes go through JPEG video mmap."""
    if _is_stackable_object_series(series):
        arr = _stack_object_series(series)
        return arr, {"kind": "stacked", "shape": list(arr.shape), "dtype": str(arr.dtype)}
    if pd.api.types.is_numeric_dtype(series):
        arr = series.to_numpy()
        return arr, {"kind": "numeric", "shape": list(arr.shape), "dtype": str(arr.dtype)}
    return None


def _arrow_is_image_type(typ) -> bool:
    """True for HF image structs / raw bytes that belong in JPEG mmap, not .npy."""
    import pyarrow as pa

    if pa.types.is_binary(typ) or pa.types.is_large_binary(typ) or pa.types.is_fixed_size_binary(typ):
        return True
    if pa.types.is_struct(typ):
        names = {field.name for field in typ}
        if names & {"bytes", "path"}:
            return True
        return any(_arrow_is_image_type(field.type) for field in typ)
    if pa.types.is_list(typ) or pa.types.is_large_list(typ) or pa.types.is_fixed_size_list(typ):
        return _arrow_is_image_type(typ.value_type)
    return False


def _mmap_parquet_columns(path: Path) -> list[str] | None:
    """Numeric / vector columns only. None = schema unavailable, read everything."""
    try:
        import pyarrow.parquet as pq

        schema = pq.ParquetFile(path).schema_arrow
    except Exception:
        return None
    keep = [field.name for field in schema if not _arrow_is_image_type(field.type)]
    return keep or None


def _read_parquet_for_mmap(path: Path, *, episode_index: int | None = None) -> pd.DataFrame:
    columns = _mmap_parquet_columns(path)
    if episode_index is None:
        return pd.read_parquet(path, columns=columns)
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(
            path,
            columns=columns,
            filters=[("episode_index", "=", int(episode_index))],
        )
        frame = table.to_pandas()
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["episode_index"] == int(episode_index)]
    if frame.empty:
        raise FileNotFoundError(f"episode {episode_index} not found in {path}")
    return frame.reset_index(drop=True)


def _write_episode_columns(df: pd.DataFrame, cache_dir: Path, *, source: str) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"source": source, "columns": {}}
    for col in df.columns:
        packed = _mmap_array_for_series(df[col])
        if packed is None:
            continue
        arr, spec = packed
        atomic_save_npy(cache_dir / f"{col}.npy", arr)
        manifest["columns"][col] = spec
    atomic_write_text(cache_dir / MANIFEST, json.dumps(manifest, indent=2))
    return manifest


def _load_episode_view(cache_dir: Path, manifest: dict[str, Any]) -> TrajectoryDataView:
    columns: dict[str, _ColumnView] = {}
    for col, spec in manifest["columns"].items():
        if spec.get("kind") == "object":
            continue
        path = cache_dir / f"{col}.npy"
        arr = np.load(path, mmap_mode="r")
        columns[col] = _ColumnView(arr, stacked=spec["kind"] == "stacked")
    return TrajectoryDataView(_columns=columns)


class MmapTrajectoryStore:
    """Build-once, mmap-many cache for LeRobot episode parquet files."""

    def __init__(self, dataset_path: Path | str, *, enabled: bool = True):
        self.dataset_path = Path(dataset_path)
        self.enabled = enabled
        self.cache_root = self.dataset_path / MMAP_DIRNAME
        self._views: dict[str, TrajectoryDataView] = {}

    def load(self, parquet_path: Path | str) -> TrajectoryDataView:
        parquet_path = Path(parquet_path)
        if not self.enabled:
            return self._load_from_parquet(parquet_path)

        key = parquet_path.resolve().as_posix()
        if key in self._views:
            return self._views[key]

        cache_dir = self._cache_dir_for(parquet_path)
        view = self._try_load_ready(cache_dir, source=str(parquet_path))
        if view is not None:
            self._views[key] = view
            return view
        with exclusive_cache_lock(cache_dir / ".lock"):
            view = self._try_load_ready(cache_dir, source=str(parquet_path))
            if view is None:
                logger.debug("Building mmap cache for %s", parquet_path.name)
                manifest = _write_episode_columns(
                    _read_parquet_for_mmap(parquet_path), cache_dir, source=str(parquet_path)
                )
                view = _load_episode_view(cache_dir, manifest)
        self._views[key] = view
        return view

    def load_episode_rows(self, parquet_path: Path | str, episode_index: int) -> TrajectoryDataView:
        parquet_path = Path(parquet_path)
        if not self.enabled:
            return self._view_from_dataframe(
                _read_parquet_for_mmap(parquet_path, episode_index=int(episode_index))
            )

        key = f"{parquet_path.resolve().as_posix()}:{episode_index}"
        if key in self._views:
            return self._views[key]

        cache_dir = self._cache_dir_for(parquet_path) / f"episode_{int(episode_index):06d}"
        source_tag = f"{parquet_path}#{episode_index}"
        view = self._try_load_ready(cache_dir, source=source_tag)
        if view is not None:
            self._views[key] = view
            return view
        with exclusive_cache_lock(cache_dir / ".lock"):
            view = self._try_load_ready(cache_dir, source=source_tag)
            if view is None:
                manifest = _write_episode_columns(
                    _read_parquet_for_mmap(parquet_path, episode_index=int(episode_index)),
                    cache_dir,
                    source=source_tag,
                )
                view = _load_episode_view(cache_dir, manifest)
        self._views[key] = view
        return view

    def prebuild_files(self, parquet_paths: list[Path | str], *, workers: int = 0) -> tuple[int, int]:
        """Build per-file parquet caches. Returns ``(built, already_present)``."""
        seen: set[str] = set()
        candidates: list[tuple[Path, Path, str]] = []
        for raw in parquet_paths:
            path = Path(raw)
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            cache_dir = self._cache_dir_for(path)
            candidates.append((path, cache_dir, str(path)))
        return self._prebuild_from_candidates(candidates, workers=workers, desc="mmap parquet", unit="file")

    def prebuild_episode_rows(
        self, items: list[tuple[Path | str, int]], *, workers: int = 0
    ) -> tuple[int, int]:
        """Build packed-parquet episode-row caches. Returns ``(built, already_present)``."""
        seen: set[str] = set()
        candidates: list[tuple[Path, Path, str, int]] = []
        for raw, episode_index in items:
            path = Path(raw)
            epi = int(episode_index)
            key = f"{path}:{epi}"
            if key in seen:
                continue
            seen.add(key)
            cache_dir = self._cache_dir_for(path) / f"episode_{epi:06d}"
            source = f"{path}#{epi}"
            candidates.append((path, cache_dir, source, epi))
        pending, ready = self._ready_by_source(candidates)
        jobs = [
            {"path": str(path), "cache_dir": str(cache_dir), "source": source, "episode_index": epi}
            for path, cache_dir, source, epi in pending
        ]
        built = _run_parquet_jobs(jobs, workers=workers, desc="mmap parquet episodes", unit="ep")
        return built, len(ready)

    def _prebuild_from_candidates(
        self,
        candidates: list[tuple[Path, Path, str]],
        *,
        workers: int,
        desc: str,
        unit: str,
    ) -> tuple[int, int]:
        pending, ready = self._ready_by_source(candidates)
        jobs = [
            {"path": str(path), "cache_dir": str(cache_dir), "source": source}
            for path, cache_dir, source in pending
        ]
        built = _run_parquet_jobs(jobs, workers=workers, desc=desc, unit=unit)
        return built, len(ready)

    def _ready_by_source(self, candidates: list) -> tuple[list, list]:
        """``candidates`` items are ``(..., cache_dir, source, ...)`` with those at [1], [2]."""
        if not candidates:
            return [], []
        if not self.cache_root.is_dir():
            return list(candidates), []
        from lbm.dataloader.custom.common.fs import map_ready

        def _hit(item) -> bool:
            return self._cache_ready(item[1], source=item[2])

        return map_ready(_hit, candidates, desc="mmap parquet ready")

    @staticmethod
    def _ready_manifest(cache_dir: Path, *, source: str) -> dict[str, Any] | None:
        manifest = read_source_manifest(cache_dir, source)
        if manifest is None:
            return None
        names = [f"{col}.npy" for col in (manifest.get("columns") or {})]
        if not cache_files_present(cache_dir, names):
            return None
        return manifest

    @staticmethod
    def _cache_ready(cache_dir: Path, *, source: str) -> bool:
        return MmapTrajectoryStore._ready_manifest(cache_dir, source=source) is not None

    @staticmethod
    def _try_load_ready(cache_dir: Path, *, source: str) -> TrajectoryDataView | None:
        """Open an already-built cache without taking the exclusive flock."""
        manifest = MmapTrajectoryStore._ready_manifest(cache_dir, source=source)
        if manifest is None:
            return None
        return _load_episode_view(cache_dir, manifest)

    def _cache_dir_for(self, parquet_path: Path) -> Path:
        rel = parquet_path.relative_to(self.dataset_path)
        safe = rel.as_posix().replace("/", "__")
        if safe.endswith(".parquet"):
            safe = safe[: -len(".parquet")]
        return self.cache_root / safe

    @staticmethod
    def _load_from_parquet(parquet_path: Path) -> TrajectoryDataView:
        return MmapTrajectoryStore._view_from_dataframe(_read_parquet_for_mmap(parquet_path))

    @staticmethod
    def _view_from_dataframe(df: pd.DataFrame) -> TrajectoryDataView:
        columns: dict[str, _ColumnView] = {}
        for col in df.columns:
            series = df[col]
            if _is_stackable_object_series(series):
                columns[col] = _ColumnView(_stack_object_series(series), stacked=True)
            elif pd.api.types.is_numeric_dtype(series):
                columns[col] = _ColumnView(series.to_numpy())
            else:
                columns[col] = _ColumnView(np.asarray(series.tolist(), dtype=object))
        return TrajectoryDataView(_columns=columns)


def _prebuild_one_parquet(job: dict) -> bool:
    path = Path(job["path"])
    cache_dir = Path(job["cache_dir"])
    source = str(job["source"])
    with exclusive_cache_lock(cache_dir / ".lock"):
        if read_source_manifest(cache_dir, source) is not None:
            return False
        _write_episode_columns(
            _read_parquet_for_mmap(path, episode_index=job.get("episode_index")),
            cache_dir,
            source=source,
        )
    return True


def _run_parquet_jobs(jobs: list[dict], *, workers: int, desc: str, unit: str) -> int:
    if not jobs:
        return 0
    from lbm.utils.progress import track

    nproc = int(workers) if workers and workers > 0 else max(1, min(32, os.cpu_count() or 8))
    nproc = max(1, min(nproc, len(jobs)))
    if nproc == 1:
        return sum(int(_prebuild_one_parquet(job)) for job in track(jobs, desc=desc, unit=unit))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as pool:
        futures = [pool.submit(_prebuild_one_parquet, job) for job in jobs]
        return sum(
            int(fut.result())
            for fut in track(as_completed(futures), total=len(futures), desc=f"{desc} x{nproc}", unit=unit)
        )
