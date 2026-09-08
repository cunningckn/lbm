"""Numpy ``.npz`` episode dumps (fallback for any spec)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import CustomSpec


def scan_numpy(root: Path, spec: CustomSpec, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    files = sorted(root.glob("*.npz")) + sorted(
        (root / "episodes").glob("*.npz") if (root / "episodes").is_dir() else []
    )
    from lbm.utils.progress import track

    records = []
    for path in track(files, desc=f"scan {spec.name}", unit="ep", leave=False):
        with np.load(path) as data:
            n = int(np.asarray(data["state"]).shape[0])
        records.append(EpisodeRecord(kind="numpy", path=str(path), n_frames=n))
        if max_episodes is not None and len(records) >= max_episodes:
            break
    return records


def read_numpy_vectors(
    record: EpisodeRecord,
    spec: CustomSpec,
    *,
    action_freq: float | None = None,
    **_kwargs,
):
    with np.load(record.path, allow_pickle=True) as data:
        state = np.asarray(data["state"], dtype=np.float32)
        if "action" in data:
            action = np.asarray(data["action"], dtype=np.float32)
        else:
            from lbm.action_space import derive_absolute_actions

            freq = spec.fps if action_freq is None else float(action_freq)
            action = derive_absolute_actions(
                state,
                spec,
                native_fps=spec.fps,
                action_freq=freq,
            )
            return fit_dim(state, spec.state_dim), action
    return fit_dim(state, spec.state_dim), fit_dim(action, spec.action_dim)


def read_numpy_frames(record: EpisodeRecord, spec: CustomSpec, cam: str, indices: list[int]):
    with np.load(record.path, allow_pickle=True) as data:
        key = f"image.{cam}"
        if key in data:
            frames = data[key]
        elif cam in data:
            frames = data[cam]
        else:
            raise KeyError(f"camera {cam!r} missing from numpy dump {record.path}")
        arr = np.asarray(frames, dtype=np.uint8)
    n = arr.shape[0]
    idx = np.clip(np.asarray(indices), 0, n - 1)
    return arr[idx]
