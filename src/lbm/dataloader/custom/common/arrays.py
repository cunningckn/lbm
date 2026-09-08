"""Shared array helpers for custom dataset readers."""

from __future__ import annotations

import numpy as np


def fit_dim(arr: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.shape[-1] == dim:
        return x
    out = np.zeros((*x.shape[:-1], dim), dtype=np.float32)
    n = min(dim, x.shape[-1])
    out[..., :n] = x[..., :n]
    return out
