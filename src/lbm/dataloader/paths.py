"""On-disk dataset root: ``<repo>/datasets/<name>`` (override with ``LBM_DATASETS``)."""

from __future__ import annotations

import os
from pathlib import Path

from lbm.config import lbm_repo_root


def datasets_root() -> Path:
    env = os.environ.get("LBM_DATASETS") or os.environ.get("lbm_DATASETS")
    if env:
        return Path(env).expanduser().resolve()
    return (lbm_repo_root() / "datasets").resolve()


def resolve_dataset(name_or_path: str, *, required: bool = True) -> Path | None:
    """Resolve a dataset name (``kai0``) or filesystem path to a directory.

    Names map to ``datasets_root() / name``. Absolute/relative paths that exist
    are returned as-is.
    """
    raw = str(name_or_path or "").strip()
    if not raw:
        if required:
            raise FileNotFoundError("empty dataset name")
        return None
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    named_bases = [datasets_root(), Path.cwd() / "datasets"]
    for base in named_bases:
        named = base / raw
        if named.exists():
            return named.resolve()
    if required:
        raise FileNotFoundError(f"dataset not found: {raw!r} (looked at {path} and {named})")
    return None
