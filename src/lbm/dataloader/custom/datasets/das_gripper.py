"""DAS gripper slim: episode.hdf5 + wrist mp4 (missing cam_high is black + mask)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lbm.action_space import dual_eef
from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import WRISTS, CustomSpec, make_spec
from lbm.dataloader.custom.video import read_mp4_indices

NAME = "das_gripper"
SPEC = make_spec("das_gripper", "das_gripper", WRISTS, 16, 16, 30.0, 13, kind="das", action_space=dual_eef())


def scan(root: Path, spec: CustomSpec, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    from lbm.dataloader.custom.common.fs import h5_nframes, list_files

    hits = list_files(Path(root), name="episode.hdf5", dir_depth=3, max_files=max_episodes, siblings=True)
    ns = h5_nframes([path for path, _names in hits], "observations/left_eef_pose", desc=f"scan {spec.name}")
    records: list[EpisodeRecord] = []
    for (h5, names), n in zip(hits, ns, strict=True):
        rec = _record(h5, spec, n, names)
        if rec is not None:
            records.append(rec)
    return records


def _record(h5: Path, spec: CustomSpec, n: int, names: frozenset[str]) -> EpisodeRecord | None:
    if n <= 0:
        return None
    parent = h5.parent
    if "cam_left_wrist.mp4" not in names:
        return None
    cam_paths = {
        cam: str(parent / fname)
        for cam, fname in (
            ("cam_left_wrist", "cam_left_wrist.mp4"),
            ("cam_right_wrist", "cam_right_wrist.mp4"),
        )
        if fname in names
    }
    return EpisodeRecord(
        kind="das",
        path=str(h5),
        n_frames=n,
        lang=_lang(parent),
        extra={"dir": str(parent), "videos": cam_paths},
    )


def _lang(parent: Path) -> str:
    try:
        return parent.parent.parent.name.replace("_", " ")
    except Exception:
        return ""


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
    fname = "cam_left_wrist.mp4" if cam == "cam_left_wrist" else "cam_right_wrist.mp4"
    path = folder / fname
    return path if path.is_file() else None


def read_frames(record: EpisodeRecord, spec: CustomSpec, cam: str, indices: list[int]) -> np.ndarray:
    hw = (spec.image_size, spec.image_size)
    raw = (record.extra or {}).get("videos", {}).get(cam)
    path = Path(raw) if raw else _cam_video_path(Path((record.extra or {}).get("dir") or ""), cam)
    if path is None:
        return np.zeros((len(indices), hw[0], hw[1], 3), dtype=np.uint8)
    return read_mp4_indices(path, indices, fallback_hw=hw)
