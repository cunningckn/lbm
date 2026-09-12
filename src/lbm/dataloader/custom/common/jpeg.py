"""JPEG bytes → RGB uint8 (zarr / Lance stills)."""

from __future__ import annotations

import numpy as np


def jpeg_bytes(cell) -> bytes | None:
    raw = cell
    for _ in range(8):
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return bytes(raw)
        if isinstance(raw, np.ndarray):
            if raw.dtype == object:
                raw = raw.reshape(-1)[0] if raw.size else None
                continue
            if raw.dtype == np.uint8:
                return raw.reshape(-1).tobytes()
            if raw.shape == ():
                raw = raw.item()
                continue
        if hasattr(raw, "item") and not isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.item()
                continue
            except (ValueError, AttributeError):
                break
        break
    return None


def still_rgb(raw, hw: tuple[int, int]) -> np.ndarray:
    blank = np.zeros(hw + (3,), dtype=np.uint8)
    if raw is None:
        return blank
    if isinstance(raw, (bytes, bytearray, memoryview)):
        blob = bytes(raw)
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[..., :3].astype(np.uint8)
        blob = jpeg_bytes(raw)
        if not blob:
            return blank
    import cv2

    bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        try:
            from io import BytesIO

            from PIL import Image

            return np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError):
            return blank
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
