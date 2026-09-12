"""DAS gripper slim: episode.hdf5 + wrist mp4 (missing cam_high is black + mask).

Listing is the six top-level ``das_gripper_slim_meta.json`` files (or a single
task folder that has one). Do not walk ``[STAGE 3]`` — 2.3M dirents, mixed depth.
A dump with no task meta falls back to a full ``episode.hdf5`` walk (unit tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lbm.action_space import dual_eef
from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import WRISTS, CustomSpec, make_spec
from lbm.dataloader.custom.video import read_mp4_indices

NAME = "das_gripper"
SPEC = make_spec(
    "das_gripper", "das_gripper", WRISTS, 16, 16, 30.0, 13, kind="das", action_space=dual_eef(format="xyz_quat")
)

_META = "das_gripper_slim_meta.json"
_H5 = "episode.hdf5"
_LEFT = "cam_left_wrist.mp4"
_RIGHT = "cam_right_wrist.mp4"
_POSE = "observations/left_eef_pose"


def scan(root: Path, spec: CustomSpec, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    from lbm.dataloader.custom.common.fs import h5_nframes, list_files

    root = Path(root)
    paths = _meta_paths(root, max_episodes)
    if not paths:
        paths = list_files(root, name=_H5, max_files=max_episodes)
    ns = h5_nframes(paths, _POSE, desc=f"scan {spec.name}")
    records: list[EpisodeRecord] = []
    for path, n in zip(paths, ns, strict=True):
        rec = _record(path, n)
        if rec is not None:
            records.append(rec)
    return records


def _meta_paths(root: Path, max_episodes: int | None) -> list[Path]:
    from lbm.dataloader.custom.common.fs import child_dirs, take

    if (root / _META).is_file():
        metas = [root / _META]
    else:
        metas = [p / _META for p in child_dirs(root) if (p / _META).is_file()]
    metas.sort(key=lambda p: str(p))
    paths: list[Path] = []
    for meta in metas:
        paths.extend(_paths_in_meta(meta))
        if max_episodes is not None and len(paths) >= max_episodes:
            break
    return take(paths, max_episodes)


def _paths_in_meta(meta: Path) -> list[Path]:
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    episodes = data.get("episodes") if isinstance(data, dict) else None
    if not isinstance(episodes, list):
        return []
    task = meta.parent
    out: list[Path] = []
    for rel in episodes:
        if not isinstance(rel, str):
            continue
        rel = rel.replace("\\", "/").lstrip("/")
        if not rel.endswith(_H5) or ".." in Path(rel).parts:
            continue
        out.append(task / rel)
    return out


def _record(h5: Path, n: int) -> EpisodeRecord | None:
    if n <= 0:
        return None
    parent = h5.parent
    left = parent / _LEFT
    if not left.is_file():
        return None
    videos = {"cam_left_wrist": str(left)}
    right = parent / _RIGHT
    if right.is_file():
        videos["cam_right_wrist"] = str(right)
    return EpisodeRecord(
        kind="das",
        path=str(h5),
        n_frames=n,
        lang=_lang(parent),
        extra={"dir": str(parent), "videos": videos},
    )


def _lang(parent: Path) -> str:
    return parent.parent.parent.name.replace("_", " ")


def read_vectors(
    record: EpisodeRecord,
    spec: CustomSpec,
    *,
    action_freq: float | None = None,
    **_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(record.path, "r") as f:
        left = np.asarray(f["observations/left_eef_pose"][...], dtype=np.float32)
        right = np.asarray(f["observations/right_eef_pose"][...], dtype=np.float32)
        mag_l = np.asarray(f["observations/mag_left"][...], dtype=np.float32).reshape(-1, 1)
        mag_r = np.asarray(f["observations/mag_right"][...], dtype=np.float32).reshape(-1, 1)
        state = np.concatenate([left, mag_l, right, mag_r], axis=1)
    state = fit_dim(state, spec.state_dim)
    from lbm.action_space import derive_absolute_actions

    freq = spec.fps if action_freq is None else float(action_freq)
    action = derive_absolute_actions(state, spec, native_fps=spec.fps, action_freq=freq)
    return state, action


def _cam_video_path(folder: Path, cam: str) -> Path | None:
    if cam == "cam_high":
        return None
    fname = _LEFT if cam == "cam_left_wrist" else _RIGHT
    path = folder / fname
    return path if path.is_file() else None


def read_frames(record: EpisodeRecord, spec: CustomSpec, cam: str, indices: list[int]) -> np.ndarray:
    hw = (spec.image_size, spec.image_size)
    raw = (record.extra or {}).get("videos", {}).get(cam)
    path = Path(raw) if raw else _cam_video_path(Path((record.extra or {}).get("dir") or ""), cam)
    if path is None:
        return np.zeros((len(indices), hw[0], hw[1], 3), dtype=np.uint8)
    return read_mp4_indices(path, indices, fallback_hw=hw)
