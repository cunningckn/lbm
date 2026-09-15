"""Small dependency boundary shared by the cache builder and reader."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
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
    if (
        not value
        or "\\" in value
        or any(ord(c) < 32 for c in value)
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or ".." in path.parts
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError(f"runtime index contains an unsafe path: {value!r}")
    return path


def write_checksum_manifest(root: Path, known_hashes: dict[str, str] | None = None) -> str:
    """Hash every delivery file, optionally reusing trusted source hashes."""

    entries: list[tuple[str, str]] = []
    excluded = {"SHA256SUMS", "PILOT_READY.json", "READY.json"}
    trusted = known_hashes or {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not portable delivery files: {path}")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            relative = path.relative_to(root).as_posix()
            digest = trusted[relative] if relative in trusted else sha256_file(path)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"invalid trusted SHA-256 for {relative}")
            entries.append((digest, relative))
    manifest = root / "SHA256SUMS"
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for digest, relative in entries:
            handle.write(f"{digest}  {relative}\n")
    return sha256_file(manifest)


def checksum_entries(root: Path) -> dict[str, str]:
    """Parse a portable manifest without reading the potentially large payloads."""
    if root.is_symlink() or (root / "SHA256SUMS").is_symlink():
        raise ValueError("symlink in manifest root")
    entries: dict[str, str] = {}
    for line_number, line in enumerate((root / "SHA256SUMS").read_text().splitlines(), 1):
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        safe_relative_path(relative)
        if relative in entries or relative in {"SHA256SUMS", "READY.json", "PILOT_READY.json"}:
            raise ValueError(f"duplicate or reserved manifest entry: {relative}")
        path = root / relative
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise ValueError(f"symlink in manifest path: {relative}")
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"missing or unsafe manifest entry: {relative}")
        entries[relative] = digest
    if not entries:
        raise ValueError("empty checksum manifest")
    return entries


def verify_checksum_manifest(root: Path, *, require_complete: bool = True, verify_files: bool = True) -> dict[str, Any]:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"checksum manifest is missing: {manifest}")
    entries = checksum_entries(root)
    if require_complete:
        excluded = {"SHA256SUMS", "PILOT_READY.json", "READY.json"}
        paths = list(root.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError("symlink in delivery")
        actual_files = {p.relative_to(root).as_posix() for p in paths if p.is_file()} - excluded
        if actual_files != set(entries):
            raise ValueError(f"manifest coverage mismatch: {sorted(actual_files ^ set(entries))[:10]}")
    for relative, digest in entries.items():
        if verify_files and sha256_file(root / relative) != digest:
            raise ValueError(f"checksum mismatch for {relative}")
    return {
        "checked_files": len(entries) if verify_files else 0,
        "listed_files": len(entries),
        "content_verified": verify_files,
        "sha256sums_sha256": sha256_file(manifest),
    }


def require_ready_cache(root: Path) -> None:
    """Check publication and schema before opening arrays; not a full payload audit."""
    metadata = read_json(root / "dataset.json")
    if metadata.get("format_version") != 1:
        raise ValueError("unsupported cache format_version")
    markers = [p for p in (root / "READY.json", root / "PILOT_READY.json") if p.is_file()]
    if len(markers) != 1:
        raise ValueError("cache must have exactly one readiness marker")
    ready = read_json(markers[0])
    if ready.get("format_version") != 1 or ready.get("status") not in {"ready", "pilot-ready"}:
        raise ValueError("unsupported readiness version or status")
    digest = ready.get("sha256sums_sha256")
    if not digest or digest != sha256_file(root / "SHA256SUMS"):
        raise ValueError("readiness manifest digest mismatch or checksums disabled")
    # Hash only the small schema here. Full payload checking is an explicit audit.
    entries = checksum_entries(root)
    verify_checksum_manifest(root, verify_files=False)
    if entries.get("dataset.json") != sha256_file(root / "dataset.json"):
        raise ValueError("dataset metadata digest mismatch")
    for relative, expected in entries.items():
        if relative.endswith(".parquet") and sha256_file(root / relative) != expected:
            raise ValueError(f"index digest mismatch: {relative}")


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
