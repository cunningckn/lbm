"""Custom dump loader. Specs live under ``custom/datasets/<name>.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lbm.dataloader.custom.dataset import CustomMixtureDataset, CustomSingleDataset, Episode
from lbm.dataloader.custom.datasets import CUSTOM_MIXTURES, CUSTOM_SPECS, uses_custom_backend
from lbm.dataloader.custom.sources import load_episodes, save_numpy_episode
from lbm.dataloader.custom.spec import CustomSpec

__all__ = [
    "CUSTOM_MIXTURES",
    "CUSTOM_SPECS",
    "CustomMixtureDataset",
    "CustomSingleDataset",
    "CustomSpec",
    "Episode",
    "get_custom_dataset",
    "load_episodes",
    "make_custom_dataset",
    "save_numpy_episode",
    "uses_custom_backend",
]


def make_custom_dataset(
    path: Path | str,
    spec_name: str,
    data_cfg: dict[str, Any],
) -> CustomSingleDataset:
    if spec_name not in CUSTOM_SPECS:
        raise KeyError(f"unknown custom spec {spec_name!r}; known: {sorted(CUSTOM_SPECS)}")
    spec = CUSTOM_SPECS[spec_name]
    from lbm.dataloader.custom.scan import scan_root
    from lbm.dataloader.paths import resolve_dataset

    root = resolve_dataset(path, required=True)
    max_episodes = data_cfg["max_episodes"] if "max_episodes" in data_cfg else None
    if max_episodes is not None:
        max_episodes = int(max_episodes)
    records = scan_root(root, spec, max_episodes=max_episodes, rescan=bool(data_cfg["rescan"]))
    kwargs: dict[str, Any] = {
        "action_length": float(data_cfg["action_length"]),
        "history_length": float(data_cfg["history_length"]),
        "use_mmap": bool(data_cfg["use_mmap"]),
        "use_mmap_frames": bool(data_cfg["use_mmap_frames"]),
        "action_mode": data_cfg["action_mode"],
        "root": root,
    }
    if "action_kind" in data_cfg and data_cfg["action_kind"]:
        kwargs["action_kind"] = data_cfg["action_kind"]
    if "action_format" in data_cfg and data_cfg["action_format"]:
        kwargs["action_format"] = data_cfg["action_format"]
    if "action_freq" in data_cfg:
        kwargs["action_freq"] = data_cfg["action_freq"]
    if "history_freq" in data_cfg:
        kwargs["history_freq"] = data_cfg["history_freq"]
    if "image_size" in data_cfg:
        kwargs["image_size"] = data_cfg["image_size"]
    if "mmap_jpeg_quality" in data_cfg:
        kwargs["mmap_jpeg_quality"] = int(data_cfg["mmap_jpeg_quality"])
    if "norm_stats" in data_cfg:
        kwargs["norm_stats"] = data_cfg["norm_stats"]
    return CustomSingleDataset(spec, records=records, **kwargs)


def get_custom_dataset(
    data_cfg: dict[str, Any],
    *,
    mode: str = "train",
    seed: int = 42,
) -> CustomMixtureDataset:
    mix_name = data_cfg["data_mix"]
    if mix_name not in CUSTOM_MIXTURES:
        raise KeyError(f"unknown custom mixture {mix_name!r}; known: {sorted(CUSTOM_MIXTURES)}")
    from lbm.dataloader.paths import datasets_root

    raw_root = data_cfg["data_root_dir"]
    root = Path(raw_root) if raw_root else datasets_root()
    pairs: list[tuple[CustomSingleDataset, float]] = []
    for folder, weight, spec_name in CUSTOM_MIXTURES[mix_name]:
        pairs.append((make_custom_dataset(root / folder, spec_name, data_cfg), float(weight)))
    return CustomMixtureDataset(pairs, mode=mode, seed=seed)
