"""Small dependency boundary shared by the cache builder and reader."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


# These values describe the on-disk cache format, not a builder implementation.
# Readers import them directly so they never need to import conversion code.
GENERIC_SUBSETS = ("egodex", "hdepic", "molmospaces", "xperience", "ytvis")
READY_MARKER = "READY.json"
PILOT_READY_MARKER = "PILOT_READY.json"


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


def completion_marker_name(*, pilot: bool) -> str:
    """Return the one marker name valid for a full result or a pilot."""

    return PILOT_READY_MARKER if pilot else READY_MARKER


def write_completion_marker(
    root: Path, payload: Mapping[str, Any], *, pilot: bool
) -> Path:
    """Write exactly one completed-result marker after all output checks pass."""

    marker = root / completion_marker_name(pilot=pilot)
    other = root / completion_marker_name(pilot=not pilot)
    if other.exists():
        raise ValueError(f"ambiguous completion markers in {root}: {marker.name}, {other.name}")
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("completion marker requires a non-empty string status")
    write_json(marker, dict(payload))
    return marker


def read_completion_marker(
    root: Path, *, pilot: bool | None = None
) -> tuple[Path, dict[str, Any]]:
    """Read a completed-result marker, rejecting partial and ambiguous outputs."""

    full_marker = root / READY_MARKER
    pilot_marker = root / PILOT_READY_MARKER
    if full_marker.is_file() and pilot_marker.is_file():
        raise ValueError(f"ambiguous completion markers in {root}: {READY_MARKER}, {PILOT_READY_MARKER}")
    if pilot is None:
        candidates = [full_marker, pilot_marker]
    else:
        candidates = [root / completion_marker_name(pilot=pilot)]
    found = [path for path in candidates if path.is_file()]
    if not found:
        expectation = (
            f"{READY_MARKER} or {PILOT_READY_MARKER}"
            if pilot is None
            else completion_marker_name(pilot=pilot)
        )
        raise FileNotFoundError(f"completed output has no {expectation}: {root}")
    if len(found) != 1:
        names = ", ".join(path.name for path in found)
        raise ValueError(f"ambiguous completion markers in {root}: {names}")
    payload = read_json(found[0])
    if not isinstance(payload, dict):
        raise ValueError(f"completion marker is not a JSON object: {found[0]}")
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"completion marker has no non-empty status: {found[0]}")
    return found[0], payload


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


def verify_completion_manifest(root: Path, readiness: Mapping[str, Any]) -> str:
    """Verify that a completion marker names the checksum manifest it finalizes."""

    expected = readiness.get("sha256sums_sha256")
    manifest = root / "SHA256SUMS"
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"completion marker has no SHA-256 manifest digest: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"completed output has no SHA256SUMS: {root}")
    actual = sha256_file(manifest)
    if actual != expected:
        raise ValueError(
            f"SHA256SUMS digest mismatch for {root}: expected {expected}, got {actual}"
        )
    return actual


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
