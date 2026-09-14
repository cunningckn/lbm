"""Small dependency boundary shared by the cache builder and reader."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required for the portable Parquet index; install the "
            "standalone package dependencies before running this command."
        ) from error
    return pa, pq


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write schema-less empty Parquet: {path}")
    pa, pq = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    _, pq = require_pyarrow()
    return pq.read_table(path).to_pylist()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_stat_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"runtime index contains an unsafe path: {value!r}")
    return path


def write_checksum_manifest(
    root: Path, known_hashes: dict[str, str] | None = None
) -> str:
    """Hash every delivery file, optionally reusing trusted source hashes."""

    entries: list[tuple[str, str]] = []
    excluded = {"SHA256SUMS", "PILOT_READY.json", "READY.json"}
    trusted = known_hashes or {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            relative = path.relative_to(root).as_posix()
            digest = trusted[relative] if relative in trusted else sha256_file(path)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"invalid trusted SHA-256 for {relative}")
            entries.append((digest, relative))
    manifest = root / "SHA256SUMS"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for digest, relative in entries:
            handle.write(f"{digest}  {relative}\n")
    return sha256_file(manifest)


def verify_checksum_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"checksum manifest is missing: {manifest}")
    checked = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            digest, separator, relative = line.partition("  ")
            if not separator or len(digest) != 64 or not relative:
                raise ValueError(f"invalid SHA256SUMS line {line_number}")
            path = root / safe_relative_path(relative)
            if not path.is_file():
                raise FileNotFoundError(f"manifest entry is missing: {relative}")
            actual = sha256_file(path)
            if actual != digest:
                raise ValueError(
                    f"checksum mismatch for {relative}: expected {digest}, got {actual}"
                )
            checked += 1
    return {"checked_files": checked, "sha256sums_sha256": sha256_file(manifest)}


def byte_size(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())


def fsync_directory(path: Path) -> None:
    """Best-effort directory sync before the final same-filesystem rename."""

    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
