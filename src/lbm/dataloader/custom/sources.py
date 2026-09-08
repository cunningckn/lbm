"""Numpy episode dumps. LeRobot trees go through ``scan_lerobot``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.dataset import Episode
from lbm.dataloader.custom.spec import CustomSpec


def save_numpy_episode(path: Path | str, episode: Episode) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray | str] = {
        "state": np.asarray(episode.state, dtype=np.float32),
        "action": np.asarray(episode.action, dtype=np.float32),
        "lang": np.asarray(episode.lang),
    }
    for cam, frames in episode.images.items():
        payload[f"image.{cam}"] = np.asarray(frames, dtype=np.uint8)
    np.savez_compressed(path, **payload)


def load_numpy_episode(path: Path | str, spec: CustomSpec) -> Episode:
    with np.load(path, allow_pickle=True) as data:
        images = {}
        for cam in spec.camera_keys:
            key = f"image.{cam}"
            if key in data:
                images[cam] = np.asarray(data[key], dtype=np.uint8)
            elif cam in data:
                images[cam] = np.asarray(data[cam], dtype=np.uint8)
        lang = data["lang"] if "lang" in data else ""
        if isinstance(lang, np.ndarray):
            lang = str(lang.item() if lang.ndim == 0 else lang.reshape(-1)[0])
        return Episode(
            images=images,
            state=fit_dim(np.asarray(data["state"], dtype=np.float32), spec.state_dim),
            action=fit_dim(np.asarray(data["action"], dtype=np.float32), spec.action_dim),
            lang=str(lang),
        )


def load_episodes(path: Path | str, spec: CustomSpec) -> list[Episode]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"custom dataset path not found: {root}")
    if (root / "meta" / "info.json").is_file():
        return _load_lerobot_episodes(root, spec)
    extra = (root / "episodes").glob("*.npz") if (root / "episodes").is_dir() else []
    npz_files = sorted(root.glob("*.npz")) + sorted(extra)
    if npz_files:
        return [load_numpy_episode(p, spec) for p in npz_files]
    raise FileNotFoundError(f"{root} is not a numpy episode dir (*.npz) or a LeRobot tree (meta/info.json)")


def _load_lerobot_episodes(root: Path, spec: CustomSpec) -> list[Episode]:
    from lbm.dataloader.custom.common.lerobot import read_lerobot_frames, read_lerobot_vectors, scan_lerobot

    records = scan_lerobot(root, spec)
    if not records:
        raise FileNotFoundError(f"no LeRobot episodes under {root}")
    episodes: list[Episode] = []
    for record in records:
        state, action = read_lerobot_vectors(record, spec)
        n = int(state.shape[0])
        images = {cam: read_lerobot_frames(record, spec, cam, list(range(n))) for cam in spec.camera_keys}
        episodes.append(Episode(images=images, state=state, action=action, lang=record.lang))
    return episodes
