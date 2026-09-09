"""Detect official vs LBM dump layouts (no scan / FK / norm)."""

from __future__ import annotations

from pathlib import Path


def _child_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def has_info(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file()


def is_lerobot(root: Path, *, max_depth: int = 3) -> bool:
    if has_info(root):
        return True
    if max_depth <= 0 or not root.is_dir():
        return False
    for child in _child_dirs(root)[:32]:
        if is_lerobot(child, max_depth=max_depth - 1):
            return True
    return False


def is_mcap(root: Path) -> bool:
    """ABC layout: data/{train,val}/<task>/episode_*/episode.mcap. Do not rglob the corpus."""
    for split in ("train", "val"):
        split_dir = root / "data" / split
        if not split_dir.is_dir():
            continue
        for task in _child_dirs(split_dir)[:8]:
            for ep in _child_dirs(task)[:4]:
                if (ep / "episode.mcap").is_file():
                    return True
    return False


def is_zarr(root: Path) -> bool:
    if not root.is_dir():
        return False
    if any(root.glob("*.zarr")):
        return True
    return any(p.is_dir() and p.suffix == ".zarr" for p in _child_dirs(root))


def is_lance(root: Path) -> bool:
    if any(p.name.startswith("table_") for p in _child_dirs(root)):
        return True
    return (root / "meta" / "hy_episodes.jsonl").is_file()


def is_hifi(root: Path) -> bool:
    return any(p.name.startswith("chunk-") for p in _child_dirs(root))


def is_agibot(root: Path) -> bool:
    if (root / "proprio_stats").is_dir():
        return True
    names = {p.name for p in _child_dirs(root)}
    if "AgiBotWorld_alpha" in names or "AgiBotWorld_beta" in names:
        return True
    for child in _child_dirs(root):
        if (child / "proprio_stats").is_dir():
            return True
    return False


def is_das_slim(root: Path) -> bool:
    if (root / "das_gripper_slim_meta.json").is_file():
        return True
    return any((p / "das_gripper_slim_meta.json").is_file() for p in _child_dirs(root))


def is_das_official_hdf5(root: Path) -> bool:
    """Official DAS-Sample-Data: episode h5 with observations/eef_pos (cameras inside)."""
    if is_das_slim(root):
        return False
    for path in _iter_h5(root, limit=8):
        if _h5_has(path, "observations/eef_pos") or _h5_has(path, "observations/cameras"):
            return True
    return False


def is_rmbench_hdf5(root: Path) -> bool:
    data = root / "data"
    if data.is_dir() and any(data.glob("*/demo_clean")):
        return True
    return any(root.glob("*/demo_clean")) or any(root.glob("data/*/demo_clean"))


def galaxea_tars(root: Path) -> list[Path]:
    hits: list[Path] = []
    for folder in (root / "lerobot", root):
        if not folder.is_dir():
            continue
        hits.extend(sorted(folder.glob("*.tar.gz")))
        hits.extend(sorted(folder.glob("*.tgz")))
    return hits


def is_galaxea_extracted(root: Path) -> bool:
    n = 0
    for child in _child_dirs(root):
        if has_info(child) or has_info(child / child.name):
            n += 1
        if n >= 1:
            return True
    return False


def _iter_h5(root: Path, *, limit: int = 32) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for path in root.rglob("*"):
        if path.suffix.lower() in {".h5", ".hdf5"} and path.is_file():
            out.append(path)
            if len(out) >= limit:
                break
    return out


def _h5_has(path: Path, key: str) -> bool:
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(path, "r") as f:
            return key in f
    except OSError:
        return False


def ready_kind(root: Path, name: str) -> str | None:
    """Return a short label if ``root`` is already the LBM dump for ``name``."""
    if not root.exists():
        return None
    if name == "abc" and is_mcap(root):
        return "mcap"
    if name == "agibot" and is_agibot(root):
        return "agibot"
    if name == "das_gripper" and is_das_slim(root):
        return "das_slim"
    if name == "droid" and is_lerobot(root):
        return "lerobot"
    if name == "egoverse" and is_zarr(root):
        return "zarr"
    if name == "galaxea" and is_galaxea_extracted(root):
        return "galaxea"
    if name == "hifi_umi" and (is_hifi(root) or is_lerobot(root)):
        return "hifi"
    if name == "hy_lance" and is_lance(root):
        return "lance"
    if name == "kai0" and is_lerobot(root):
        return "lerobot"
    if name == "libero" and is_lerobot(root):
        return "lerobot"
    if name == "rmbench" and is_lerobot(root) and not is_rmbench_hdf5(root):
        return "lerobot"
    if name == "robotwin" and is_lerobot(root):
        return "lerobot"
    return None
