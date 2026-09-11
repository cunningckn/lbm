"""One module per named dump: ``NAME``, ``SPEC``, ``scan`` / ``read_vectors`` / ``read_frames``."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType

from .mixes import NAMED_MIXES


def _load_modules() -> dict[str, ModuleType]:
    out: dict[str, ModuleType] = {}
    pkg_dir = str(Path(__file__).resolve().parent)
    for info in pkgutil.iter_modules([pkg_dir]):
        if info.name.startswith("_") or info.name == "mixes":
            continue
        mod = importlib.import_module(f"{__package__}.{info.name}")
        name = getattr(mod, "NAME", None)
        if name and hasattr(mod, "SPEC"):
            out[str(name)] = mod
    return out


MODULES = _load_modules()
CUSTOM_SPECS = {name: mod.SPEC for name, mod in MODULES.items()}
CUSTOM_MIXTURES: dict[str, list[tuple[str, float, str]]] = {name: [(name, 1.0, name)] for name in CUSTOM_SPECS}
CUSTOM_MIXTURES.update({name: list(rows) for name, rows in NAMED_MIXES.items()})


def module_for(name: str) -> ModuleType:
    if name not in MODULES:
        raise KeyError(f"unknown custom dataset {name!r}; known: {sorted(MODULES)}")
    return MODULES[name]


def uses_custom_backend(*, robot_type: str = "", data_mix: str = "", dataset: str = "") -> bool:
    """True for a named spec, ``all``, or a comma mix of specs."""
    if data_mix:
        if data_mix in CUSTOM_MIXTURES:
            return True
        if "," in data_mix:
            parts = [p.strip() for p in data_mix.split(",") if p.strip()]
            return bool(parts) and all(p in CUSTOM_SPECS for p in parts)
        return False
    if robot_type and robot_type in CUSTOM_SPECS:
        return True
    name = str(dataset).rstrip("/").rsplit("/", 1)[-1]
    return bool(name) and name in CUSTOM_SPECS


def dataset_names() -> tuple[str, ...]:
    """Return registered dataset names in stable order."""
    return tuple(sorted(MODULES))


def dataset_spec(name: str):
    """Resolve one registered dataset's immutable IO specification."""
    return module_for(name).SPEC


def dataset_module(name: str) -> ModuleType:
    """Resolve one registered dataset adapter module."""
    return module_for(name)


def dataset_mixes() -> dict[str, list[tuple[str, float, str]]]:
    """Return a copy of named mixtures so callers cannot mutate the registry."""
    return {name: list(rows) for name, rows in CUSTOM_MIXTURES.items()}
