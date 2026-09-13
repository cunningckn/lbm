"""Causal history sampling in seconds, with explicit validity and bounded storage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from lbm.temporal import n_steps

MAX_HISTORY_STEPS = 64


@dataclass(frozen=True)
class HistorySelection:
    indices: np.ndarray
    valid: np.ndarray
    offsets: np.ndarray


def history_offsets(length: float, frequency: float) -> np.ndarray:
    if not np.isfinite(length) or length < 0 or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("history length must be finite/nonnegative and frequency finite/positive")
    if length * frequency > MAX_HISTORY_STEPS:
        raise ValueError(f"history may request at most {MAX_HISTORY_STEPS} observations")
    return np.arange(-(n_steps(length, frequency) - 1), 1, dtype=np.float64) / frequency


def frame_history(frame: int, *, lower: int, fps: float, length: float, frequency: float) -> HistorySelection:
    """Sample native frames at or before target times, without an episode-sized index."""
    if not 0 <= lower <= frame or not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid physical history bounds or frame rate")
    targets = frame + history_offsets(length, frequency) * fps
    indices = np.floor(targets + 1e-9).astype(np.int64)
    valid = (indices >= lower) & (targets - indices <= fps / frequency + 1e-9)
    indices = np.maximum(indices, lower)
    return HistorySelection(indices, valid, (indices - frame) / fps)


def timestamp_history(timestamps, *, now: float, length: float, frequency: float) -> HistorySelection:
    """Use the latest observation no later than each target; never interpolate from the future.

    A target without an observation within one requested sampling period is
    invalid. Padding selects the oldest observation but retains a false mask.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or not np.isfinite(now):
        raise ValueError("history timestamps must be a nonempty finite vector")
    if np.any(np.diff(times) <= 0) or times[-1] > now:
        raise ValueError("history timestamps must be strictly increasing and not in the future")
    # Subtract the origin first; epoch timestamps have much coarser ULPs than
    # short simulation clocks. Allow only their floating-point rounding error.
    tolerance = max(1e-9, 2 * abs(np.spacing(max(abs(now), np.max(np.abs(times))))))
    relative = times - now
    targets = history_offsets(length, frequency)
    indices = np.searchsorted(relative, targets + tolerance, side="right") - 1
    valid = indices >= 0
    indices = np.maximum(indices, 0)
    valid &= targets - relative[indices] <= 1 / frequency + tolerance
    return HistorySelection(indices, valid, relative[indices])


class ObservationHistory:
    """Bounded, context-scoped observations. Callers own/copy the stored values."""

    def __init__(self, *, horizon: float, capacity: int = MAX_HISTORY_STEPS):
        if not np.isfinite(horizon) or horizon < 0 or type(capacity) is not int or capacity < 1:
            raise ValueError("history requires finite nonnegative horizon and positive integer capacity")
        self.horizon = horizon
        self._items: deque[tuple[float, Any]] = deque(maxlen=capacity)
        self._context = None

    def reset(self):
        self._items.clear()
        self._context = None

    def append(self, timestamp: float, value: Any, *, context=None):
        """Append an owned snapshot; return whether a context change cleared history."""
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("observation timestamp must be finite seconds")
        changed = bool(self._items) and context != self._context
        if changed:
            self.reset()
        self._context = context
        if self._items:
            previous = self._items[-1][0]
            if timestamp < previous:
                raise ValueError("observation timestamps decreased; reset at episode boundaries")
            if timestamp == previous:
                self._items.pop()
        self._items.append((timestamp, value))
        # Keep one observation at/before the oldest target, subject to capacity.
        while len(self._items) > 1 and self._items[1][0] <= timestamp - self.horizon:
            self._items.popleft()
        return changed

    def select(self, *, length: float, frequency: float):
        if not self._items:
            raise ValueError("cannot sample empty observation history")
        times, values = zip(*self._items, strict=True)
        selection = timestamp_history(times, now=times[-1], length=length, frequency=frequency)
        return [values[i] for i in selection.indices], selection

    def __len__(self):
        return len(self._items)


HISTORY_TENSOR_FIELDS = (
    "history_mask",
    "history_offsets",
    "state_history",
    "state_history_mask",
    "state_history_offsets",
)


def collate_history(samples, out):
    """Right-align time and pad heterogeneous state widths; reject mixed contracts."""
    import torch

    for key in HISTORY_TENSOR_FIELDS:
        present = [key in sample for sample in samples]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"history batch mixes samples with and without {key}")
        arrays = [np.asarray(sample[key]) for sample in samples]
        size = max(len(array) for array in arrays)
        if key == "state_history":
            if any(array.ndim != 2 for array in arrays):
                raise ValueError("state_history samples must be (time, state_dim)")
            width = max(array.shape[1] for array in arrays)
            value = np.zeros((len(arrays), size, width), dtype=np.float32)
            for i, array in enumerate(arrays):
                value[i, size - len(array) :, : array.shape[1]] = array
        else:
            if any(array.ndim != 1 for array in arrays):
                raise ValueError(f"{key} samples must be time vectors")
            value = np.zeros((len(arrays), size), dtype=bool if key.endswith("mask") else np.float32)
            for i, array in enumerate(arrays):
                value[i, size - len(array) :] = array
        out[key] = torch.from_numpy(value)
