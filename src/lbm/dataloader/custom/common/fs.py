"""Walk a dump and read HDF5 lengths. Hidden (``.*``) names are always skipped."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

FileHit = tuple[Path, frozenset[str]]


def is_hidden(name: str) -> bool:
    return name.startswith(".")


def dir_names(folder: Path | str) -> frozenset[str]:
    """One ``scandir``; empty if the directory cannot be read."""
    return frozenset(e.name for e in _scandir(folder))


def child_dirs(root: Path | str) -> list[Path]:
    """Immediate subdirectories."""
    return [Path(e.path) for e in _scandir(root) if e.is_dir(follow_symlinks=True)]


def take(items: Iterable[T], n: int | None) -> list[T]:
    if n is None:
        return list(items)
    out: list[T] = []
    for item in items:
        out.append(item)
        if len(out) >= n:
            break
    return out


def list_files(
    root: Path,
    *,
    suffixes: tuple[str, ...] = (),
    name: str | None = None,
    dir_depth: int | None = None,
    max_files: int | None = None,
    siblings: bool = False,
    desc: str = "",
) -> list[Path] | list[FileHit]:
    """Files under ``root``.

    ``dir_depth=N`` means the file sits in a folder N levels below ``root``
    (``3`` → ``root/a/b/c/file``). Omit it to walk the whole tree.
    A full known-depth walk expands each level in parallel; ``max_files``
    stays sequential so it can stop early.
    ``siblings=True`` returns ``(path, sibling_names)`` from the same listing.
    """
    if name is None and not suffixes:
        raise ValueError("list_files requires name= or suffixes=")
    root = Path(root)
    suffix = tuple(suffixes)
    if max_files is None and dir_depth is not None and dir_depth >= 2:
        hits = _hits_at(root, suffix, name, dir_depth)
    else:
        it: Iterable[FileHit] = _iter_hits(root, suffix, name, dir_depth)
        if desc:
            from lbm.utils.progress import track

            it = track(it, desc=desc, unit="ep", leave=False)
        hits = take(it, max_files)
    if siblings:
        return hits
    return [path for path, _names in hits]


def list_dirs(
    root: Path,
    *,
    suffix: str = "",
    max_depth: int = 1,
    max_dirs: int | None = None,
) -> list[Path]:
    """Directories whose name ends with ``suffix`` (``.zarr``), depth ``1..max_depth``.

    Does not descend into a matched directory. Full walks expand each level in
    parallel; ``max_dirs`` stays sequential so it can stop early.
    """
    root = Path(root)
    if max_dirs is not None:
        return take(_iter_dirs(root, suffix, max_depth), max_dirs)
    found: list[Path] = []
    layer = [root]
    for _ in range(max(1, int(max_depth))):
        nxt = _expand(layer)
        if suffix:
            found.extend(p for p in nxt if p.name.endswith(suffix))
            layer = [p for p in nxt if not p.name.endswith(suffix)]
        else:
            found.extend(nxt)
            layer = nxt
        if not layer:
            break
    found.sort(key=lambda p: str(p))
    return found


def h5_nrows(job: tuple[str, str]) -> int:
    """``(path, dataset)`` → first-axis length, or 0. Safe inside a forked worker."""
    import h5py

    path, dataset = job
    try:
        handle = h5py.File(path, "r", locking=False, rdcc_nbytes=0)
    except TypeError:
        handle = h5py.File(path, "r")
    except OSError:
        return 0
    try:
        ds = handle.get(dataset)
        return 0 if ds is None else int(ds.shape[0])
    except OSError:
        return 0
    finally:
        handle.close()


def h5_nframes(paths: list[Path] | list[str], dataset: str, *, desc: str = "") -> list[int]:
    return map_proc(h5_nrows, [(str(p), dataset) for p in paths], desc=desc)


def map_threads(
    func: Callable[[T], R],
    items: list[T],
    *,
    desc: str = "",
    workers: int = 0,
    chunksize: int = 1,
) -> list[R]:
    """Thread-map ``func``. For metadata / non-HDF5 I/O only."""
    return _map(func, items, desc=desc, workers=workers, chunksize=chunksize, processes=False)


def map_proc(
    func: Callable[[T], R],
    items: list[T],
    *,
    desc: str = "",
    workers: int = 0,
    chunksize: int = 0,
) -> list[R]:
    """Process-map ``func``. Uses ``fork`` so workers do not re-import ``__main__``."""
    return _map(func, items, desc=desc, workers=workers, chunksize=chunksize, processes=True, min_items=64)


def map_ready(
    func: Callable[[T], bool],
    items: list[T],
    *,
    desc: str = "",
    workers: int = 0,
) -> tuple[list[T], list[T]]:
    """Thread-map a ready-check. Returns ``(pending, ready)``."""
    flags = map_threads(func, items, desc=desc, workers=workers)
    pending: list[T] = []
    ready: list[T] = []
    for item, ok in zip(items, flags, strict=True):
        (ready if ok else pending).append(item)
    return pending, ready


def _map(
    func: Callable[[T], R],
    items: list[T],
    *,
    desc: str,
    workers: int,
    chunksize: int,
    processes: bool,
    min_items: int = 1,
) -> list[R]:
    if not items:
        return []
    nproc = _nproc(items, workers, 32)
    if nproc == 1 or len(items) < min_items:
        return [func(item) for item in items]
    nchunk = int(chunksize) if chunksize else (max(8, len(items) // (nproc * 4)) if processes else 1)
    nchunk = max(1, nchunk)
    if processes:
        import multiprocessing
        import sys
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        start = "fork" if sys.platform != "win32" else "spawn"
        try:
            ctx = multiprocessing.get_context(start)
            with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as pool:
                return _consume(pool.map(func, items, chunksize=nchunk), desc, len(items))
        except (OSError, RuntimeError, BrokenProcessPool):
            return [func(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=nproc) as pool:
        return _consume(pool.map(func, items, chunksize=nchunk), desc, len(items))


def _consume(mapped, desc: str, total: int):
    if not desc:
        return list(mapped)
    from lbm.utils.progress import track

    return list(track(mapped, desc=desc, total=total, unit="ep", leave=False))


def _nproc(items: list, workers: int, cap: int) -> int:
    want = int(workers) if workers else min(cap, os.cpu_count() or 4)
    return max(1, min(want, len(items)))


def _scandir(path: Path | str) -> list[os.DirEntry]:
    try:
        with os.scandir(path) as it:
            return [e for e in it if not is_hidden(e.name)]
    except OSError:
        return []


def _matches(filename: str, suffix: tuple[str, ...], name: str | None) -> bool:
    return filename == name if name is not None else Path(filename).suffix in suffix


def _expand(layer: list[Path]) -> list[Path]:
    return [path for chunk in map_threads(child_dirs, layer) for path in chunk]


def _hits_in(folder: Path, suffix: tuple[str, ...], name: str | None) -> list[FileHit]:
    names = dir_names(folder)
    return [(folder / fname, names) for fname in names if _matches(fname, suffix, name)]


def _hits_at(root: Path, suffix: tuple[str, ...], name: str | None, dir_depth: int) -> list[FileHit]:
    folders = [root]
    for _ in range(dir_depth):
        folders = _expand(folders)
        if not folders:
            return []

    def _one(folder: Path) -> list[FileHit]:
        return _hits_in(folder, suffix, name)

    hits = [hit for chunk in map_threads(_one, folders) for hit in chunk]
    hits.sort(key=lambda item: str(item[0]))
    return hits


def _iter_hits(
    root: Path, suffix: tuple[str, ...], name: str | None, dir_depth: int | None
) -> Iterator[FileHit]:
    if dir_depth is None:
        yield from _walk_any(root, suffix, name)
        return
    if dir_depth < 1 or not root.is_dir():
        return
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        kids = sorted(child_dirs(cur), key=lambda p: p.name)
        if depth + 1 == dir_depth:
            for child in kids:
                yield from _hits_in(child, suffix, name)
            continue
        for child in reversed(kids):
            stack.append((child, depth + 1))


def _walk_any(root: Path, suffix: tuple[str, ...], name: str | None) -> Iterator[FileHit]:
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not is_hidden(d))
        names = frozenset(f for f in filenames if not is_hidden(f))
        for fname in sorted(names):
            if _matches(fname, suffix, name):
                yield Path(dirpath) / fname, names


def _iter_dirs(root: Path, suffix: str, max_depth: int) -> Iterator[Path]:
    if max_depth < 1 or not root.is_dir():
        return
    kids = sorted(child_dirs(root), key=lambda p: p.name)
    if suffix:
        leaves = [p for p in kids if p.name.endswith(suffix)]
        nested = [p for p in kids if not p.name.endswith(suffix)]
    else:
        leaves = kids
        nested = kids
    yield from leaves
    if max_depth <= 1:
        return
    for child in nested:
        yield from _iter_dirs(child, suffix, max_depth - 1)
