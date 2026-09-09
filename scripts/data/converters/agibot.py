"""Nest Alpha/Beta HF snapshots as ``AgiBotWorld_{alpha,beta}/``."""

from __future__ import annotations

from pathlib import Path

from .detect import is_agibot
from .layout import ensure_dest


def layout_agibot(src: Path, dest: Path, *, force: bool = False) -> Path:
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    alpha = _find_release(src, ("alpha", "AgiBotWorld_alpha", "AgiBotWorld-Alpha"))
    beta = _find_release(src, ("beta", "AgiBotWorld_beta", "AgiBotWorld-Beta"))
    if alpha is None and beta is None:
        if (src / "proprio_stats").is_dir():
            ensure_dest(src, dest / "AgiBotWorld_alpha", force=force)
            return dest
        raise FileNotFoundError(f"no AgiBot Alpha/Beta trees under {src}")
    if alpha is not None:
        ensure_dest(alpha, dest / "AgiBotWorld_alpha", force=force)
    if beta is not None:
        ensure_dest(beta, dest / "AgiBotWorld_beta", force=force)
    if not is_agibot(dest):
        raise RuntimeError(f"agibot layout failed at {dest}")
    return dest


def _find_release(src: Path, names: tuple[str, ...]) -> Path | None:
    if (src / "proprio_stats").is_dir() and any(n == "alpha" or n.endswith("_alpha") for n in names):
        return src
    for name in names:
        cand = src / name
        if _is_release(cand):
            return cand
    if not src.is_dir():
        return None
    for child in src.iterdir():
        if child.is_dir() and child.name in names and _is_release(child):
            return child
    return None


def _is_release(path: Path) -> bool:
    return path.is_dir() and ((path / "proprio_stats").is_dir() or (path / "observations").is_dir())
