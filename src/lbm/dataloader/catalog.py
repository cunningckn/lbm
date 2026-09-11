"""Named dumps under ``lbm/datasets/<name>``.

Every catalog entry is a custom spec. ``--dataset kai0`` resolves that folder.
``--data-mix all`` is defined in ``custom/datasets/mixes.py``; comma lists
(``kai0,libero``) concat those embodiment folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lbm.dataloader.custom.datasets import dataset_names
from lbm.dataloader.custom.datasets.mixes import NAMED_MIXES

CATALOG_CONCAT_MIXES = frozenset(NAMED_MIXES)


@dataclass(frozen=True)
class DatasetEntry:
    """One folder under ``datasets/``."""

    name: str
    backend: str = "custom"
    robot_type: str = ""
    folder: str = ""

    def __post_init__(self) -> None:
        if not self.folder:
            object.__setattr__(self, "folder", self.name)
        if not self.robot_type:
            object.__setattr__(self, "robot_type", self.name)


DATASETS: dict[str, DatasetEntry] = {name: DatasetEntry(name=name) for name in dataset_names()}


def lookup(name: str) -> DatasetEntry | None:
    key = str(name or "").strip()
    if not key:
        return None
    if key in DATASETS:
        return DATASETS[key]
    folder = key.rstrip("/").rsplit("/", 1)[-1]
    return DATASETS.get(folder)


def catalog_names(*, backend: str = "") -> tuple[str, ...]:
    if not backend or backend == "custom":
        return tuple(DATASETS)
    return ()


def select_dumps(raw: str = "", *, default: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Comma list of catalog names. Empty → ``default`` (the script's ``DATASETS``)."""
    text = str(raw or "").strip()
    if not text:
        if not default:
            raise ValueError("empty dump list; pass --dataset or a default DATASETS tuple")
        return default
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise KeyError(f"unknown dump {unknown}; known: {sorted(DATASETS)}")
    return names


def on_disk_dumps(names: tuple[str, ...], *, base: Path | None = None) -> list[tuple[str, Path]]:
    """``(spec_name, resolved_path)`` for folders that exist under ``base``."""
    from lbm.dataloader.paths import datasets_root

    root = Path(base) if base is not None else datasets_root()
    found: list[tuple[str, Path]] = []
    for name in names:
        path = root / name
        if path.is_dir():
            found.append((name, path.resolve()))
        else:
            print(f"skip {name}: not a directory at {path}", flush=True)
    return found


def parse_mix(data_mix: str) -> list[DatasetEntry] | None:
    """Expand ``all`` (from ``mixes.py``) / ``kai0,galaxea`` into catalog entries.

    Empty strings and unknown names return ``None``. A single catalog name
    such as ``robotwin`` is that embodiment folder under ``datasets/``.
    """
    raw = str(data_mix or "").strip()
    if not raw:
        return None
    if raw in NAMED_MIXES:
        return [
            DatasetEntry(name=folder, robot_type=spec) for folder, _weight, spec in NAMED_MIXES[raw]
        ]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        entry = lookup(parts[0])
        return [entry] if entry is not None else None
    out: list[DatasetEntry] = []
    unknown: list[str] = []
    for token in parts:
        found = lookup(token)
        if found is None:
            unknown.append(token)
        else:
            out.append(found)
    if unknown:
        raise KeyError(f"unknown mix part(s) {unknown}; catalog: {sorted(DATASETS)}")
    return out


def is_catalog_concat(data_mix: str) -> bool:
    """True for ``all`` / comma lists of catalog names."""
    raw = str(data_mix or "").strip()
    if raw in CATALOG_CONCAT_MIXES or "," in raw:
        return True
    return False


def mix_backends(entries: list[DatasetEntry]) -> set[str]:
    return {entry.backend for entry in entries}


def backend_for(*, dataset: str = "", robot_type: str = "", data_mix: str = "") -> str:
    """``custom`` if the name is a known spec/mix, else ``""``."""
    if data_mix:
        try:
            plan = parse_mix(data_mix)
        except KeyError:
            plan = None
        if plan:
            return "custom"
        from lbm.dataloader.custom.spec import CUSTOM_MIXTURES

        if data_mix in CUSTOM_MIXTURES:
            return "custom"
        return ""
    entry = lookup(robot_type) or lookup(dataset)
    if entry is not None or robot_type in DATASETS:
        return "custom"
    return ""
