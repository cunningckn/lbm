"""Convert (length in seconds, frequency in Hz) to native-fps delta indices.

Action windows look forward from the current step; history windows look
backward and include the current frame. Number of samples is
``round(length * freq)`` (at least 1); native stride is
``round(native_fps / freq)`` (at least 1).

Action abs/rel/delta and missing-label derivation live in ``lbm.action_space``.
"""

from __future__ import annotations

import numpy as np


def n_steps(length_s: float, freq_hz: float) -> int:
    """How many samples fit in ``length_s`` seconds at ``freq_hz``."""
    if length_s <= 0 or freq_hz <= 0:
        return 1
    return max(1, int(round(float(length_s) * float(freq_hz))))


def action_hop_frames(length_s: float, freq_hz: float, native_fps: float) -> int:
    """Native-fps span of one action window (used for computed ``delta``)."""
    idxs = delta_indices(length_s, freq_hz, native_fps, past=False)
    hop = int(idxs[-1])
    if hop > 0:
        return hop
    return native_stride(native_fps, freq_hz)


def native_stride(native_fps: float, freq_hz: float) -> int:
    """Step between samples, in native-fps frames."""
    if native_fps <= 0:
        raise ValueError(f"native_fps must be positive, got {native_fps}")
    if freq_hz <= 0:
        return 1
    return max(1, int(round(float(native_fps) / float(freq_hz))))


def delta_indices(
    length_s: float,
    freq_hz: float,
    native_fps: float,
    *,
    past: bool = False,
) -> np.ndarray:
    """Delta indices in native frames relative to the current step.

    Future (actions): ``[0, stride, 2*stride, …]``.
    Past (history): ``[-(n-1)*stride, …, -stride, 0]``.
    """
    n = n_steps(length_s, freq_hz)
    stride = native_stride(native_fps, freq_hz)
    if past:
        return np.arange(-(n - 1) * stride, 1, stride, dtype=np.int64)
    return np.arange(0, n * stride, stride, dtype=np.int64)
