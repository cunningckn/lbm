"""Random-access helpers for uncompressed tar shards.

MolmoMotion ships hundreds of thousands of small NPZ/JSON/MP4 members inside
plain ``.tar`` files.  The conversion workers index the tar headers once, then
read members by byte offset.  This avoids extracting a loose-file tree and lets
independent output shards be produced safely in parallel.
"""

from __future__ import annotations

import io
import importlib
import sys
import tarfile
from atexit import register as register_atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TarMemberRef:
    """Portable reference to one regular member of an uncompressed tar."""

    tar_relpath: str
    member_name: str
    offset: int
    size: int


def _safe_member_name(name: str) -> str:
    normalized = name.removeprefix("./")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe tar member path: {name!r}")
    return path.as_posix()


def _index_one_tar(
    root_text: str,
    tar_text: str,
    wanted_names: frozenset[str] | None,
) -> tuple[str, list[TarMemberRef]]:
    root = Path(root_text)
    path = Path(tar_text)
    relative = path.relative_to(root).as_posix()
    rows: list[TarMemberRef] = []
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = _safe_member_name(member.name)
            if wanted_names is not None and name not in wanted_names:
                continue
            rows.append(
                TarMemberRef(
                    tar_relpath=relative,
                    member_name=name,
                    offset=int(member.offset_data),
                    size=int(member.size),
                )
            )
    return relative, rows


def build_tar_index(
    source_root: str | Path,
    tar_paths: Iterable[Path],
    *,
    workers: int,
    wanted_names: set[str] | None = None,
) -> dict[str, TarMemberRef]:
    """Index regular tar members, rejecting duplicate or missing wanted names."""

    root = Path(source_root).resolve()
    paths = sorted({path.resolve() for path in tar_paths})
    if not paths:
        raise FileNotFoundError("no tar shards matched the requested source component")
    for path in paths:
        if path.parent == root or root not in path.parents:
            raise ValueError(f"tar shard is outside source root: {path}")
        if path.suffix != ".tar":
            raise ValueError(f"only uncompressed .tar shards support byte offsets: {path}")

    wanted = None if wanted_names is None else frozenset(_safe_member_name(v) for v in wanted_names)
    index: dict[str, TarMemberRef] = {}
    pool_size = max(1, min(int(workers), len(paths)))

    def merge_rows(relative: str, rows: list[TarMemberRef]) -> None:
        for row in rows:
            previous = index.get(row.member_name)
            if previous is not None:
                raise ValueError(
                    f"duplicate member {row.member_name!r} in "
                    f"{previous.tar_relpath} and {relative}"
                )
            index[row.member_name] = row

    if pool_size == 1:
        for path in paths:
            relative, rows = _index_one_tar(str(root), str(path), wanted)
            merge_rows(relative, rows)
    else:
        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            futures = {
                executor.submit(_index_one_tar, str(root), str(path), wanted): path
                for path in paths
            }
            for future in as_completed(futures):
                relative, rows = future.result()
                merge_rows(relative, rows)

    if wanted is not None:
        missing = sorted(wanted.difference(index))
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} required tar members are missing; first entries: {preview}"
            )
    return index


_OPEN_FILES: dict[str, object] = {}


def close_cached_archives() -> None:
    """Close process-local tar descriptors after a build or raw benchmark."""

    for handle in _OPEN_FILES.values():
        if not getattr(handle, "closed", True):
            handle.close()
    _OPEN_FILES.clear()


register_atexit(close_cached_archives)


def _install_numpy_pickle_compatibility() -> None:
    """Allow NumPy 1.x to unpickle object arrays written by NumPy 2.x."""

    try:
        importlib.import_module("numpy._core")
        return
    except ModuleNotFoundError:
        pass

    core = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", core)
    for child in ("multiarray", "numeric", "_multiarray_umath"):
        module = importlib.import_module(f"numpy.core.{child}")
        sys.modules.setdefault(f"numpy._core.{child}", module)


def read_member(source_root: str | Path, reference: TarMemberRef) -> bytes:
    """Read a member directly by offset, caching tar descriptors per process."""

    root = Path(source_root).resolve()
    path = (root / reference.tar_relpath).resolve()
    if path.parent == root or root not in path.parents:
        raise ValueError(f"tar reference escapes source root: {reference.tar_relpath!r}")
    key = str(path)
    handle = _OPEN_FILES.get(key)
    if handle is None or getattr(handle, "closed", True):
        handle = path.open("rb")
        _OPEN_FILES[key] = handle
    handle.seek(reference.offset)
    payload = handle.read(reference.size)
    if len(payload) != reference.size:
        raise EOFError(
            f"short tar member read for {reference.member_name}: "
            f"expected {reference.size}, got {len(payload)}"
        )
    return payload


def read_npz(source_root: str | Path, reference: TarMemberRef) -> dict[str, np.ndarray]:
    """Materialize one NPZ member without creating a loose source file."""

    payload = read_member(source_root, reference)
    _install_numpy_pickle_compatibility()
    with np.load(io.BytesIO(payload), allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}
