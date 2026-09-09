"""Extract Galaxea ``lerobot/<task>.tar.gz`` into one LeRobot folder per task."""

from __future__ import annotations

import tarfile
from pathlib import Path

from .detect import galaxea_tars, has_info, is_galaxea_extracted


def extract_galaxea(src: Path, dest: Path, *, force: bool = False) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tars = galaxea_tars(src)
    if not tars and is_galaxea_extracted(src):
        return dest if dest.resolve() == src.resolve() else _link_or_same(src, dest)
    if not tars:
        raise FileNotFoundError(f"no lerobot/*.tar.gz under {src}")
    for tar_path in tars:
        _extract_one(tar_path, dest, force=force)
    _flatten_nested(dest)
    return dest


def _link_or_same(src: Path, dest: Path) -> Path:
    from .layout import ensure_dest

    return ensure_dest(src, dest)


def _extract_one(tar_path: Path, dest: Path, *, force: bool) -> None:
    stem = tar_path.name.removesuffix(".tar.gz").removesuffix(".tgz")
    task_dir = dest / stem
    if has_info(task_dir) and not force:
        print(f"skip existing {task_dir.name}")
        return
    print(f"extract {tar_path.name}")
    task_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        names = [n for n in tar.getnames() if n and n != "."]
        top = {n.split("/")[0] for n in names}
        target = dest if stem in top else task_dir
        tar.extractall(path=target, filter="data")
    nested = dest / stem / stem
    if has_info(nested) and not has_info(task_dir):
        _replace_dir(task_dir, nested)


def _flatten_nested(dest: Path) -> None:
    for child in sorted(p for p in dest.iterdir() if p.is_dir()):
        nested = child / child.name
        if has_info(nested) and not has_info(child):
            _replace_dir(child, nested)


def _replace_dir(parent: Path, nested: Path) -> None:
    import shutil

    tmp = parent.with_name(parent.name + ".__flat__")
    nested.rename(tmp)
    shutil.rmtree(parent)
    tmp.rename(parent)


