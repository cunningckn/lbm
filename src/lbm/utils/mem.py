"""Per-worker RSS reclaim. Used by mmap prebuild processes."""

from __future__ import annotations

import gc
import os
from pathlib import Path

_LIMIT: int | None = None
_DEFAULT_RSS = 4 * 1024 * 1024 * 1024
_CGROUP_FRACTION = 0.8
_MIN_RSS = 256 * 1024 * 1024


def process_rss_bytes() -> int:
    """Current process RSS. ``/proc`` first; ``ru_maxrss`` fallback."""
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            resident = int(fh.read().split()[1])
        return resident * os.sysconf("SC_PAGESIZE")
    except (OSError, IndexError, ValueError):
        pass
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


def cgroup_memory_max_bytes() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            text = path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if not text or text == "max":
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value >= (1 << 62):
            continue
        return value
    return None


def worker_rss_limit_bytes(nproc: int | None = None) -> int:
    """Per-worker RSS budget. ``LBM_WORKER_RSS_MB`` overrides the cgroup split."""
    env = os.environ.get("LBM_WORKER_RSS_MB")
    if env:
        return max(1, int(float(env))) * 1024 * 1024
    n = int(nproc) if nproc and int(nproc) > 0 else _nproc_hint()
    cap = cgroup_memory_max_bytes()
    if cap is not None:
        return max(_MIN_RSS, int(cap * _CGROUP_FRACTION / max(n, 1)))
    return _DEFAULT_RSS


def _nproc_hint() -> int:
    for key in ("LBM_PREBUILD_WORKERS",):
        raw = os.environ.get(key)
        if raw and int(raw) > 0:
            return int(raw)
    return max(1, min(32, os.cpu_count() or 8))


def set_worker_rss_limit(limit_bytes: int | None) -> None:
    """Pin the process-local budget used by ``reclaim_if_over()``."""
    global _LIMIT
    _LIMIT = int(limit_bytes) if limit_bytes else None


def malloc_trim() -> bool:
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return True
    except Exception:
        return False


def reclaim_if_over(limit_bytes: int | None = None) -> bool:
    """If RSS exceeds the worker budget, ``gc.collect`` + ``malloc_trim``. True if it ran."""
    if limit_bytes is not None:
        limit = int(limit_bytes)
    elif _LIMIT is not None:
        limit = int(_LIMIT)
    else:
        limit = worker_rss_limit_bytes()
    if process_rss_bytes() <= limit:
        return False
    gc.collect()
    malloc_trim()
    return True
