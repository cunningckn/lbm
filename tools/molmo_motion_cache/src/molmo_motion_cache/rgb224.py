"""Bounded mmap access to pre-materialized RGB224 JPEG payloads.

The cache format is deliberately small and independent from the LBM package:
``frames.npy`` is either an ``(N, 3)`` integer matrix or a structured array
with ``shard``, ``offset``, and ``length`` columns.  The three values locate a
JPEG in ``frames-00000.bin``-style payload shards.  The reader keeps only a
bounded number of payload maps open and exposes a seek/read fallback for
parity checks.
"""

from __future__ import annotations

import io
import math
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Literal

import numpy as np


FrameReadMode = Literal["mmap", "seek"]
_INDEX_COLUMNS = ("shard", "offset", "length")
_INDEX_CHUNK_ROWS = 65_536
_MAX_INT64 = int(np.iinfo(np.int64).max)


def _close_mapping(value: object | None) -> None:
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _decode_pillow_rgb(blob: bytes | memoryview) -> np.ndarray:
    """Decode through Pillow without claiming that the JPEG decode is zero-copy."""

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific.
        raise RuntimeError("RGB224 decoding requires Pillow; install molmo-motion-cache with Pillow") from error
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _metric(samples: list[float], elapsed: float, checksum: int) -> dict[str, float | int]:
    count = len(samples)
    return {
        "samples": count,
        "elapsed_seconds": elapsed,
        "samples_per_second": count / elapsed if elapsed else math.inf,
        "latency_ms_p50": float(np.percentile(samples, 50)),
        "latency_ms_p95": float(np.percentile(samples, 95)),
        "latency_ms_mean": float(np.mean(samples)),
        "checksum": checksum,
    }


class Rgb224FrameReader:
    """Read existing indexed JPEG shards without importing the LBM training code.

    A reader is process-local.  When a PyTorch worker forks or spawns, inherited
    mappings are discarded and the small index is reopened in that worker.  A
    JPEG memoryview is available only inside :meth:`borrow_jpeg`; it is released
    before an LRU eviction can close the backing mmap.
    """

    def __init__(
        self,
        payload_root: str | Path,
        *,
        max_open_shards: int = 8,
        expected_size: int | None = 224,
    ) -> None:
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        if expected_size is not None and expected_size <= 0:
            raise ValueError("expected_size must be positive or None")
        self.root = Path(payload_root).resolve()
        self.max_open_shards = int(max_open_shards)
        self.expected_size = expected_size
        self._lock = RLock()
        self._owner_pid: int | None = os.getpid()
        self._mappings: OrderedDict[int, np.memmap] = OrderedDict()
        self._leases: dict[int, int] = {}
        self._closed = False
        self._index: np.ndarray | None = None
        self._shards: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._lengths: np.ndarray | None = None
        self._load_index()

    def __enter__(self) -> Rgb224FrameReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Make spawned DataLoader workers reopen their own maps and index."""

        state = dict(self.__dict__)
        state.update(
            {
                "_lock": None,
                "_owner_pid": None,
                "_mappings": OrderedDict(),
                "_leases": {},
                "_index": None,
                "_shards": None,
                "_offsets": None,
                "_lengths": None,
            }
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = RLock()
        self._owner_pid = os.getpid()
        self._mappings = OrderedDict()
        self._leases = {}
        if not self._closed:
            self._load_index()

    def _load_index(self) -> None:
        path = self.root / "frames.npy"
        index = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(index, np.ndarray):
            raise ValueError(f"RGB224 frame index is not an ndarray: {path}")
        if index.dtype.names is None:
            if index.ndim != 2 or index.shape[1] != len(_INDEX_COLUMNS):
                raise ValueError(
                    "frames.npy must be an (N, 3) integer matrix ordered as "
                    "(shard, offset, length)"
                )
            columns = tuple(index[:, position] for position in range(len(_INDEX_COLUMNS)))
        else:
            missing = [name for name in _INDEX_COLUMNS if name not in index.dtype.names]
            if missing:
                raise ValueError(f"frames.npy is missing structured columns: {missing}")
            if index.ndim != 1:
                raise ValueError("structured frames.npy must have one dimension")
            columns = tuple(index[name] for name in _INDEX_COLUMNS)
        if any(not np.issubdtype(column.dtype, np.integer) for column in columns):
            raise ValueError("RGB224 frame index columns must use integer dtypes")
        self._index = index
        self._shards, self._offsets, self._lengths = columns

    def _ensure_current_process(self) -> None:
        if self._closed:
            raise RuntimeError("RGB224 frame reader is closed")
        if self._owner_pid == os.getpid():
            return
        # Do not reuse inherited mmap objects or locks inside a DataLoader child.
        self._mappings = OrderedDict()
        self._leases = {}
        self._lock = RLock()
        self._owner_pid = os.getpid()
        self._index = None
        self._shards = self._offsets = self._lengths = None
        self._load_index()

    def _columns(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._shards is None or self._offsets is None or self._lengths is None:
            raise RuntimeError("RGB224 frame reader has no open index")
        return self._shards, self._offsets, self._lengths

    def __len__(self) -> int:
        self._ensure_current_process()
        shards, _offsets, _lengths = self._columns()
        return len(shards)

    @property
    def open_shards(self) -> tuple[int, ...]:
        self._ensure_current_process()
        with self._lock:
            return tuple(self._mappings)

    def _frame_parts(self, frame_index: int) -> tuple[int, int, int]:
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
            raise TypeError("frame_index must be an integer")
        index = int(frame_index)
        shards, offsets, lengths = self._columns()
        if index < 0 or index >= len(shards):
            raise IndexError(f"frame index out of range: {index}")
        shard = int(shards[index])
        offset = int(offsets[index])
        length = int(lengths[index])
        if shard < 0 or offset < 0 or length <= 0:
            raise ValueError(f"invalid RGB224 frame index row {index}")
        if offset > _MAX_INT64 - length:
            raise ValueError(f"RGB224 frame offset overflows row {index}")
        return shard, offset, length

    def _shard_path(self, shard: int) -> Path:
        padded = self.root / f"frames-{shard:05d}.bin"
        if padded.is_file():
            return padded
        unpadded = self.root / f"frames-{shard}.bin"
        if unpadded.is_file():
            return unpadded
        raise FileNotFoundError(f"RGB224 payload shard {shard} is missing below {self.root}")

    @staticmethod
    def _check_range(path: Path, offset: int, length: int) -> None:
        if offset + length > path.stat().st_size:
            raise EOFError(f"RGB224 frame range exceeds payload shard: {path}")

    def _evict_idle_locked(self) -> None:
        while len(self._mappings) > self.max_open_shards:
            for shard, mapping in tuple(self._mappings.items()):
                if self._leases.get(shard, 0) == 0:
                    del self._mappings[shard]
                    _close_mapping(mapping)
                    break
            else:
                return

    def _mapping_locked(self, shard: int, offset: int, length: int) -> np.memmap:
        mapping = self._mappings.get(shard)
        if mapping is not None:
            self._mappings.move_to_end(shard)
            return mapping
        path = self._shard_path(shard)
        self._check_range(path, offset, length)
        mapping = np.memmap(path, dtype=np.uint8, mode="r")
        self._mappings[shard] = mapping
        self._evict_idle_locked()
        return mapping

    def _release_lease(self, shard: int) -> None:
        with self._lock:
            active = self._leases.get(shard, 0)
            if active <= 1:
                self._leases.pop(shard, None)
            else:
                self._leases[shard] = active - 1
            self._evict_idle_locked()

    @contextmanager
    def borrow_jpeg(self, frame_index: int) -> Iterator[memoryview]:
        """Yield a JPEG memoryview and release it before payload-map eviction."""

        self._ensure_current_process()
        shard, offset, length = self._frame_parts(frame_index)
        with self._lock:
            self._leases[shard] = self._leases.get(shard, 0) + 1
            try:
                mapping = self._mapping_locked(shard, offset, length)
                view = memoryview(mapping)[offset : offset + length]
            except BaseException:
                self._release_lease(shard)
                raise
        try:
            yield view
        finally:
            try:
                view.release()
            except ValueError:
                pass
            self._release_lease(shard)

    def read_jpeg_bytes(self, frame_index: int) -> bytes:
        """Use the retained seek/read path for parity checks and rollback."""

        self._ensure_current_process()
        shard, offset, length = self._frame_parts(frame_index)
        path = self._shard_path(shard)
        self._check_range(path, offset, length)
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise EOFError(f"short RGB224 payload read from {path}")
        return payload

    def decode_rgb(self, frame_index: int, *, mode: FrameReadMode = "mmap") -> np.ndarray:
        """Decode one existing JPEG without changing its 224x224 preprocessing."""

        if mode == "mmap":
            with self.borrow_jpeg(frame_index) as blob:
                decoded = _decode_pillow_rgb(blob)
        elif mode == "seek":
            decoded = _decode_pillow_rgb(self.read_jpeg_bytes(frame_index))
        else:
            raise ValueError(f"unsupported RGB224 read mode: {mode!r}")
        if self.expected_size is not None and decoded.shape != (
            self.expected_size,
            self.expected_size,
            3,
        ):
            raise ValueError(
                f"RGB224 frame {frame_index} decoded to {decoded.shape}, expected "
                f"({self.expected_size}, {self.expected_size}, 3)"
            )
        return decoded

    def decode_many(
        self, frame_indices: np.ndarray | list[int], *, mode: FrameReadMode = "mmap"
    ) -> np.ndarray:
        indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if not len(indices):
            side = 0 if self.expected_size is None else self.expected_size
            return np.empty((0, side, side, 3), dtype=np.uint8)
        return np.stack([self.decode_rgb(int(index), mode=mode) for index in indices])

    def validate(self) -> dict[str, int | str]:
        """Validate every index range without decoding or copying JPEG payloads."""

        self._ensure_current_process()
        shards, offsets, lengths = self._columns()
        max_end_by_shard: dict[int, int] = {}
        for start in range(0, len(shards), _INDEX_CHUNK_ROWS):
            stop = min(start + _INDEX_CHUNK_ROWS, len(shards))
            chunk_shards = np.asarray(shards[start:stop], dtype=np.int64)
            chunk_offsets = np.asarray(offsets[start:stop], dtype=np.int64)
            chunk_lengths = np.asarray(lengths[start:stop], dtype=np.int64)
            if (
                np.any(chunk_shards < 0)
                or np.any(chunk_offsets < 0)
                or np.any(chunk_lengths <= 0)
                or np.any(chunk_offsets > _MAX_INT64 - chunk_lengths)
            ):
                raise ValueError(f"invalid RGB224 frame index rows {start}:{stop}")
            ends = chunk_offsets + chunk_lengths
            for shard in np.unique(chunk_shards):
                selector = chunk_shards == shard
                maximum = int(np.max(ends[selector]))
                max_end_by_shard[int(shard)] = max(max_end_by_shard.get(int(shard), 0), maximum)
        payload_bytes = 0
        for shard, maximum in sorted(max_end_by_shard.items()):
            path = self._shard_path(shard)
            size = path.stat().st_size
            if maximum > size:
                raise EOFError(f"RGB224 index exceeds payload shard {path}")
            payload_bytes += size
        return {
            "status": "passed",
            "frames": len(shards),
            "payload_shards": len(max_end_by_shard),
            "payload_bytes": payload_bytes,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._ensure_current_process()
        with self._lock:
            active = sum(self._leases.values())
            if active:
                raise RuntimeError("release all RGB224 JPEG views before closing the reader")
            for mapping in self._mappings.values():
                _close_mapping(mapping)
            self._mappings.clear()
            _close_mapping(self._index)
            self._index = None
            self._shards = self._offsets = self._lengths = None
            self._closed = True


def validate_rgb224_payload(
    payload_root: str | Path,
    *,
    max_open_shards: int = 8,
) -> dict[str, int | str]:
    """Standalone integrity entry point for the pre-materialized RGB payload."""

    with Rgb224FrameReader(payload_root, max_open_shards=max_open_shards) as reader:
        return reader.validate()


def benchmark_rgb224_payload(
    payload_root: str | Path,
    *,
    samples: int,
    warmup: int,
    max_open_shards: int,
    seed: int,
    expected_size: int | None = 224,
) -> dict[str, Any]:
    """Compare seek/read and mmap with identical JPEG bytes and Pillow decoding."""

    if samples <= 0 or warmup < 0:
        raise ValueError("samples must be positive and warmup cannot be negative")
    started = time.perf_counter()
    reader = Rgb224FrameReader(
        payload_root,
        max_open_shards=max_open_shards,
        expected_size=expected_size,
    )
    reader_open_seconds = time.perf_counter() - started
    try:
        if not len(reader):
            raise ValueError("cannot benchmark an empty RGB224 frame payload")
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(reader), size=samples + warmup, dtype=np.int64)
        # Parity is deliberately outside timed measurements.
        for index in indices.tolist():
            seek_blob = reader.read_jpeg_bytes(index)
            with reader.borrow_jpeg(index) as mmap_blob:
                if seek_blob != mmap_blob:
                    raise ValueError(f"RGB224 JPEG bytes differ for frame {index}")
            seek_rgb = reader.decode_rgb(index, mode="seek")
            mmap_rgb = reader.decode_rgb(index, mode="mmap")
            if not np.array_equal(seek_rgb, mmap_rgb):
                raise ValueError(f"RGB224 Pillow decode differs for frame {index}")

        for index in indices[:warmup].tolist():
            reader.decode_rgb(index, mode="seek")
        for index in indices[:warmup].tolist():
            reader.decode_rgb(index, mode="mmap")

        def measure(mode: FrameReadMode) -> dict[str, float | int]:
            latencies: list[float] = []
            checksum = 0
            began = time.perf_counter()
            for index in indices[warmup:].tolist():
                tick = time.perf_counter()
                image = reader.decode_rgb(index, mode=mode)
                checksum += int(image.sum(dtype=np.uint64))
                latencies.append((time.perf_counter() - tick) * 1000.0)
            return _metric(latencies, time.perf_counter() - began, checksum)

        seek = measure("seek")
        mmap = measure("mmap")
        if seek["checksum"] != mmap["checksum"]:
            raise ValueError("RGB224 benchmark seek/mmap checksums differ")
        return {
            "status": "completed",
            "payload_root": str(Path(payload_root).resolve()),
            "parameters": {
                "samples": samples,
                "warmup": warmup,
                "max_open_shards": max_open_shards,
                "seed": seed,
                "expected_size": expected_size,
            },
            "reader_open_seconds": reader_open_seconds,
            "seek_read_pillow": seek,
            "mmap_memoryview_pillow": mmap,
            "throughput_speedup": mmap["samples_per_second"] / seek["samples_per_second"],
            "checksums_equal": True,
            "limitations": [
                "Both paths use Pillow; this measures payload access plus decode, not GPU transfer.",
                "Pillow receives a BytesIO wrapper, so mmap avoids seek/read but does not claim zero-copy JPEG decode.",
                "OS page-cache state is not cleared and should be reported with any production result.",
            ],
        }
    finally:
        reader.close()
