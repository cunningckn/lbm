"""Place a processed dump at ``datasets/<name>`` (symlink when possible)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def ensure_dest(src: Path, dest: Path, *, force: bool = False) -> Path:
    src = src.resolve()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.resolve() == src:
            return dest
        if not force:
            return dest
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    try:
        os.symlink(src, dest)
        print(f"linked {dest} -> {src}")
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
        print(f"copied {src} -> {dest}")
    return dest
