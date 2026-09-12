"""JPEG encode/decode and letterbox resize."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

# Pin native thread pools before libomp/OpenCV init. Python threads + OpenMP
# (cv2.imdecode) otherwise deadlocks — the hang when running decode in a pool.
for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except (AttributeError, cv2.error):
    pass

JpegBlob = bytes | memoryview
_IMREAD_RGB = getattr(cv2, "IMREAD_COLOR_RGB", None)


def decode_thread_init() -> None:
    """Initializer for JPEG ThreadPoolExecutor workers."""
    cv2.setNumThreads(1)


def decode_threads_for_loaders(n_loaders: int, *, cpus: int | None = None, cap: int | None = None) -> int:
    """Keep total JPEG threads near CPU count: ``cpus // num_workers``.

    Many DataLoader workers cap at 4 threads each even on large ``cpu_count``:
    16×8 decode pools oversubscribe memory bandwidth (warm p50 decode ~16ms
    vs ~6ms in a single-process microbench).
    """
    n_cpu = int(cpus) if cpus is not None else (os.cpu_count() or 8)
    n_loaders = max(1, int(n_loaders))
    if cap is None:
        cap = 4 if n_loaders >= 8 else 8
    return max(1, min(int(cap), n_cpu // n_loaders))


def letterbox_content_hw(height: int, width: int, image_size: int) -> tuple[int, int]:
    """Content size ``resize_with_pad`` scales to before padding."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    scale = min(image_size / float(width), image_size / float(height))
    nw = max(1, int(round(width * scale)))
    nh = max(1, int(round(height * scale)))
    return nh, nw


def resize_with_pad(image: np.ndarray, image_size: int, pad_value: int = 0) -> np.ndarray:
    """Letterbox HWC image to a square."""
    h, w = image.shape[:2]
    if (h, w) == (image_size, image_size):
        return image
    nh, nw = letterbox_content_hw(h, w, image_size)
    if (h, w) != (nh, nw):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size, image_size, image.shape[2]), pad_value, dtype=np.uint8)
    y0 = (image_size - nh) // 2
    x0 = (image_size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = image
    return canvas


def encode_jpeg(image_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def encode_jpeg_rgb(frame: np.ndarray, *, quality: int = 85) -> bytes:
    """Encode an RGB uint8 frame (H, W, 3) to JPEG bytes."""
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected RGB frame (H, W, 3), got {frame.shape}")
    return encode_jpeg(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), quality=quality)


def _decode_jpeg_bgr(data: JpegBlob) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img


def _decode_jpeg_into(data: JpegBlob, dest: np.ndarray) -> None:
    """Decode one JPEG as RGB into ``dest`` (H, W, 3)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if _IMREAD_RGB is not None:
        img = cv2.imdecode(arr, _IMREAD_RGB)
        if img is None:
            raise RuntimeError("cv2.imdecode failed")
        dest[...] = img
        return
    cv2.cvtColor(_decode_jpeg_bgr(data), cv2.COLOR_BGR2RGB, dst=dest)


def decode_still_rgb(blob: JpegBlob) -> np.ndarray:
    """Decode JPEG or PNG bytes to RGB uint8 (H, W, 3)."""
    return decode_jpeg_rgb(blob)


def decode_jpeg_rgb(blob: JpegBlob) -> np.ndarray:
    """Decode JPEG bytes to an RGB uint8 frame (H, W, 3)."""
    arr = np.frombuffer(blob, dtype=np.uint8)
    if _IMREAD_RGB is not None:
        img = cv2.imdecode(arr, _IMREAD_RGB)
        if img is None:
            raise RuntimeError("cv2.imdecode failed")
        return img
    return cv2.cvtColor(_decode_jpeg_bgr(blob), cv2.COLOR_BGR2RGB)


def decode_jpegs_into(
    blobs: Sequence[JpegBlob],
    out: np.ndarray,
    executor: ThreadPoolExecutor | None = None,
) -> np.ndarray:
    """Decode JPEGs as RGB into a preallocated ``(N, H, W, 3)`` array.

    Thread-pool work is one contiguous slice per worker (not one task per JPEG)
    so queue overhead stays small at history=18.
    """
    n = len(blobs)
    if out.ndim != 4 or out.shape[-1] != 3:
        raise ValueError(f"expected dest (N, H, W, 3), got {out.shape}")
    if n != int(out.shape[0]):
        raise ValueError(f"blob count {n} != dest N {out.shape[0]}")

    def _fill_slice(bounds: tuple[int, int]) -> None:
        start, end = bounds
        for i in range(start, end):
            _decode_jpeg_into(blobs[i], out[i])

    if executor is None or n <= 1:
        _fill_slice((0, n))
        return out
    n_chunks = min(int(getattr(executor, "_max_workers", 1) or 1), n)
    chunk = (n + n_chunks - 1) // n_chunks
    slices = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    list(executor.map(_fill_slice, slices))
    return out
