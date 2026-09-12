"""Exact float32 quantiles using a disk spool and bounded radix-selection buffers."""

from __future__ import annotations

import tempfile

import numpy as np


class DiskQuantiles:
    """Four sequential byte-histogram passes, without sorting or mapping the full dataset."""

    def __init__(self, *, directory=None, block_rows: int = 65536):
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        self.block_rows = block_rows
        self.file = tempfile.TemporaryFile(dir=directory)
        self.rows = 0
        self.dim = None

    def close(self):
        self.file.close()

    def append(self, values):
        values = np.asarray(values, dtype=np.float32)
        dim = values.shape[1]
        if self.dim is not None and dim != self.dim:
            raise ValueError("quantile dimension changed")
        self.file.seek(0, 2)
        self.file.write(values.tobytes(order="C"))
        self.dim = dim
        self.rows += len(values)

    def quantiles(self, probabilities):
        if not self.rows:
            raise ValueError("no quantile observations")
        p = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("quantile probabilities must be in [0, 1]")
        positions = p * (self.rows - 1)
        lower, upper = np.floor(positions).astype(np.int64), np.ceil(positions).astype(np.int64)
        ranks = np.concatenate((lower, upper))
        remaining = np.broadcast_to(ranks[:, None], (len(ranks), self.dim)).copy()
        prefixes = np.zeros(remaining.shape, dtype=np.uint32)
        self.file.flush()
        for shift in (24, 16, 8, 0):
            hist = np.zeros((*remaining.shape, 256), dtype=np.int64)
            self.file.seek(0)
            for start in range(0, self.rows, self.block_rows):
                count = min(self.block_rows, self.rows - start) * self.dim
                values = np.fromfile(self.file, dtype=np.float32, count=count).reshape(-1, self.dim)
                if values.size != count:
                    raise OSError("truncated quantile spool")
                bits = values.view(np.uint32)
                keys = bits ^ np.where(bits >> 31, np.uint32(0xffffffff), np.uint32(0x80000000))
                for d in range(self.dim):
                    column = keys[:, d]
                    byte = ((column >> shift) & 255).astype(np.int64)
                    for k in range(len(ranks)):
                        selected = byte if shift == 24 else byte[column >> (shift + 8) == prefixes[k, d]]
                        hist[k, d] += np.bincount(selected, minlength=256)
            totals = hist.cumsum(axis=-1)
            chosen = (totals <= remaining[..., None]).sum(axis=-1)
            previous = np.take_along_axis(totals, np.maximum(chosen - 1, 0)[..., None], axis=-1)[..., 0]
            remaining -= np.where(chosen > 0, previous, 0)
            prefixes = (prefixes << 8) | chosen.astype(np.uint32)
        bits = prefixes ^ np.where(prefixes >> 31, np.uint32(0x80000000), np.uint32(0xffffffff))
        values = bits.view(np.float32).astype(np.float64)
        low, high = np.split(values, 2)
        return low + (high - low) * (positions - lower)[:, None]
