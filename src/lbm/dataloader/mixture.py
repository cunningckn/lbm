"""Concatenate catalog dumps under ``datasets/<name>``; collate pads IO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lbm.config import TrainConfig, data_cfg_from_train
from lbm.dataloader.catalog import DatasetEntry
from lbm.dataloader.custom.dataset import CustomMixtureDataset


def build_catalog_mixture(
    entries: list[DatasetEntry],
    config: TrainConfig,
    *,
    mode: str,
    skip_missing: bool = False,
) -> CustomMixtureDataset:
    """Load each catalog entry from ``datasets/<name>`` and concat."""
    from lbm.dataloader.custom import make_custom_dataset
    from lbm.dataloader.paths import datasets_root, resolve_dataset

    data_cfg = data_cfg_from_train(config)

    pairs: list[tuple[Any, float]] = []
    missing: list[str] = []
    root = Path(config.data.data_root_dir).expanduser() if config.data.data_root_dir else datasets_root()
    for entry in entries:
        if config.data.data_root_dir:
            candidate = root / entry.name
            path = candidate.resolve() if candidate.is_dir() else None
        else:
            path = resolve_dataset(entry.name, required=False)
        if path is None:
            missing.append(entry.name)
            print(f"[mix] skip {entry.name}: not on disk (tried {root / entry.name})", flush=True)
            continue
        pairs.append((make_custom_dataset(path, entry.robot_type, data_cfg), 1.0))
    if missing and not skip_missing:
        raise FileNotFoundError(f"mix members not on disk: {missing}")
    if missing:
        print(f"[mix] skipped {len(missing)} missing, loaded {len(pairs)}/{len(entries)}", flush=True)
    if not pairs:
        raise FileNotFoundError(f"mix has no on-disk datasets (missing {missing})")
    return CustomMixtureDataset(pairs, mode=mode, seed=config.seed)


def load_selected_dumps(
    config: TrainConfig,
    names: tuple[str, ...],
    *,
    mode: str = "train",
) -> CustomMixtureDataset:
    """Load each catalog name from ``data_root/<name>`` with spec ``name`` (no robot-type inference)."""
    from lbm.dataloader.catalog import on_disk_dumps
    from lbm.dataloader.custom import make_custom_dataset
    from lbm.dataloader.paths import datasets_root

    data_cfg = data_cfg_from_train(config)
    base = Path(config.data.data_root_dir) if config.data.data_root_dir else datasets_root()
    pairs: list[tuple[Any, float]] = []
    for name, path in on_disk_dumps(names, base=base):
        pairs.append((make_custom_dataset(path, name, data_cfg), 1.0))
    if not pairs:
        raise FileNotFoundError(f"no on-disk dumps in {base} from {list(names)}")
    return CustomMixtureDataset(pairs, mode=mode, seed=config.seed)


def load_named_or_mix(
    config: TrainConfig,
    *,
    mode: str = "train",
    dataset_path: str = "",
) -> CustomMixtureDataset:
    """Load ``--dataset`` as one dump, or ``--data-mix`` as a named folder mix."""
    from lbm.dataloader.catalog import lookup
    from lbm.dataloader.custom import (
        CUSTOM_SPECS,
        get_custom_dataset,
        make_custom_dataset,
    )
    from lbm.dataloader.paths import datasets_root, resolve_dataset

    data_cfg = data_cfg_from_train(config)
    raw = dataset_path or config.data.dataset or config.data.robot_type
    robot_type = config.data.robot_type
    entry = lookup(raw) or lookup(robot_type)
    if entry is not None and not robot_type:
        robot_type = entry.robot_type
    path = resolve_dataset(raw, required=False) if raw else None
    if path is not None and path.is_dir() and robot_type in CUSTOM_SPECS:
        single = make_custom_dataset(path, robot_type, data_cfg)
        return CustomMixtureDataset([(single, 1.0)], mode=mode, seed=config.seed)
    if path is not None and path.is_dir():
        raise ValueError(
            f"unknown spec {robot_type!r} for {path}; "
            f"pass a catalog name (--dataset kai0) or --robot-type; known: {sorted(CUSTOM_SPECS)}"
        )
    if not config.data.data_mix:
        raise ValueError("set TrainConfig.data.dataset or data.data_mix")
    data_cfg["data_root_dir"] = config.data.data_root_dir or str(datasets_root())
    data_cfg["data_mix"] = config.data.data_mix
    return get_custom_dataset(data_cfg, mode=mode, seed=config.seed)


def load_dataset(
    config: TrainConfig,
    *,
    mode: str = "train",
    dataset_path: str = "",
) -> CustomMixtureDataset:
    """Same dump resolution as training: catalog concat, named mix, or one folder."""
    from lbm.dataloader.catalog import CATALOG_CONCAT_MIXES, is_catalog_concat, parse_mix

    mix_name = str(config.data.data_mix or "").strip()
    plan = parse_mix(mix_name) if mix_name and not dataset_path else None
    if plan and is_catalog_concat(mix_name):
        return build_catalog_mixture(
            plan,
            config,
            mode=mode,
            skip_missing=mix_name in CATALOG_CONCAT_MIXES,
        )
    return load_named_or_mix(config, mode=mode, dataset_path=dataset_path)
