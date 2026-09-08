"""Per-worker RSS reclaim for mmap prebuild processes."""

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
        if value < (1 << 62):
            return value
    return None


def worker_rss_limit_bytes(nproc: int | None = None) -> int:
    """Per-worker RSS budget. ``LBM_WORKER_RSS_MB`` overrides the cgroup split."""
    env = os.environ.get("LBM_WORKER_RSS_MB")
    if env:
        return max(1, int(float(env))) * 1024 * 1024
    n = int(nproc) if nproc and int(nproc) > 0 else _nproc_hint()
    cap = cgroup_memory_max_bytes()
    if cap is None:
        return _DEFAULT_RSS
    return max(_MIN_RSS, int(cap * _CGROUP_FRACTION / max(n, 1)))


def _nproc_hint() -> int:
    raw = os.environ.get("LBM_PREBUILD_WORKERS")
    if raw and int(raw) > 0:
        return int(raw)
    return max(1, min(32, os.cpu_count() or 8))


def set_worker_rss_limit(limit_bytes: int | None) -> None:
    """Pin the process-local budget. ``None`` falls back to ``worker_rss_limit_bytes()``."""
    global _LIMIT
    _LIMIT = None if limit_bytes is None else int(limit_bytes)


def malloc_trim() -> bool:
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return True
    except Exception:
        return False


def _active_limit(limit_bytes: int | None) -> int:
    if limit_bytes is not None:
        return int(limit_bytes)
    if _LIMIT is not None:
        return _LIMIT
    return worker_rss_limit_bytes()


def reclaim_if_over(limit_bytes: int | None = None) -> bool:
    """If RSS exceeds the budget, ``gc.collect`` + ``malloc_trim``. True if it ran."""
    if process_rss_bytes() <= _active_limit(limit_bytes):
        return False
    gc.collect()
    malloc_trim()
    return True
