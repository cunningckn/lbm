"""AgiBotWorld: HDF5 proprio + per-camera mp4."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from lbm.action_space import ABS, DELTA, GRIPPER, JOINT, QUANTILE, ActionSlice
from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import CustomSpec, make_spec
from lbm.dataloader.custom.video import read_mp4_indices

_AGIBOT_SPACE = (
    ActionSlice("left_arm", 0, 7, DELTA, JOINT, QUANTILE),
    ActionSlice("right_arm", 7, 14, DELTA, JOINT, QUANTILE),
    ActionSlice("left_gripper", 14, 15, ABS, GRIPPER, QUANTILE),
    ActionSlice("right_gripper", 15, 16, ABS, GRIPPER, QUANTILE),
    ActionSlice("head", 16, 18, DELTA, JOINT, QUANTILE),
    ActionSlice("waist", 18, 20, DELTA, JOINT, QUANTILE),
    ActionSlice("base", 20, 22, ABS, JOINT, QUANTILE, state_start=0, state_end=0),
)

NAME = "agibot"
SPEC = make_spec(
    "agibot",
    "agibot_genie1",
    ("top_head", "hand_left", "hand_right"),
    20,
    22,
    30.0,
    26,
    kind="agibot",
    action_space=_AGIBOT_SPACE,
    scan_revision=2,
)

_CAM = {"top_head": "head", "hand_left": "hand_left", "hand_right": "hand_right"}


def _h5(name: str, f, default_t: int, width: int) -> np.ndarray:
    if name not in f:
        return np.zeros((default_t, width), dtype=np.float32)
    arr = np.asarray(f[name][...], dtype=np.float32)
    if arr.size == 0:
        return np.zeros((default_t, width), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def scan(root: Path, spec: CustomSpec, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    from lbm.dataloader.custom.common.fs import h5_nframes

    pairs = _list_h5(Path(root), max_episodes)
    ns = h5_nframes([path for _base, path in pairs], "state/joint/position", desc=f"scan {spec.name}")
    videos = _list_videos(pairs, spec.camera_keys)
    return _records(pairs, ns, videos, spec)


def _list_h5(root: Path, max_episodes: int | None) -> list[tuple[Path, Path]]:
    from lbm.dataloader.custom.common.fs import list_files

    pairs: list[tuple[Path, Path]] = []
    remaining = max_episodes
    for base in _bases(root):
        paths = list_files(_stats_root(base), suffixes=(".h5", ".hdf5"), dir_depth=2, max_files=remaining)
        pairs.extend((base, path) for path in paths)
        if remaining is not None:
            remaining -= len(paths)
            if remaining <= 0:
                break
    return pairs


def _list_videos(pairs: list[tuple[Path, Path]], camera_keys: tuple[str, ...]) -> list[dict[str, str]]:
    from lbm.dataloader.custom.common.fs import map_threads

    folders = [_video_dir(base, h5) for base, h5 in pairs]
    return map_threads(lambda folder: _cam_paths(folder, camera_keys), folders)


def _records(
    pairs: list[tuple[Path, Path]],
    ns: list[int],
    videos: list[dict[str, str]],
    spec: CustomSpec,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    langs: dict[tuple[str, str], dict[str, str]] = {}
    provenance: dict[tuple[str, str], dict[str, str | None]] = {}
    for (base, h5), n, cams in zip(pairs, ns, videos, strict=True):
        key = (str(base), h5.parent.parent.name)
        rec = _record(base, h5, spec, n, langs, cams, provenance.setdefault(key, {}))
        if rec is not None:
            records.append(rec)
    return records


def _bases(root: Path) -> list[Path]:
    from lbm.dataloader.custom.common.fs import dir_names

    if (root / "proprio_stats").is_dir():
        return [root]
    kids = [root / name for name in sorted(dir_names(root)) if (root / name / "proprio_stats").is_dir()]
    return kids or ([root] if root.is_dir() else [])


def _stats_root(base: Path) -> Path:
    stats = base / "proprio_stats"
    return stats if stats.is_dir() else base


def _video_dir(base: Path, h5: Path) -> Path:
    return base / "observations" / h5.parent.parent.name / h5.parent.name / "videos"


def _record(
    base: Path,
    h5: Path,
    spec: CustomSpec,
    n: int,
    langs: dict[tuple[str, str], dict[str, str]],
    videos: dict[str, str],
    annotation_sources: dict[str, str | None],
) -> EpisodeRecord | None:
    if n <= 0:
        return None
    episode_id = str(int(h5.parent.name))
    task_id = h5.parent.parent.name
    video_dir = _video_dir(base, h5)
    key = (str(base), task_id)
    if key not in langs:
        langs[key] = _instructions(base, task_id, sources=annotation_sources)
    return EpisodeRecord(
        kind="agibot",
        path=str(h5),
        n_frames=n,
        lang=langs[key].get(episode_id, langs[key].get("*", "")),
        extra={
            "annotation_sources": annotation_sources,
            "parent_id": f"agibot:{task_id}:{episode_id}",
            "video_dir": str(video_dir),
            "task_id": task_id,
            "episode_id": episode_id,
            "videos": videos,
        },
    )


def _cam_paths(video_dir: Path, camera_keys: tuple[str, ...]) -> dict[str, str]:
    from lbm.dataloader.custom.common.fs import dir_names

    names = dir_names(video_dir)
    out: dict[str, str] = {}
    for cam in camera_keys:
        prefix = _CAM.get(cam, cam)
        for suffix in ("_color.mp4", "_color_h264.mp4", ".mp4"):
            fname = f"{prefix}{suffix}"
            if fname in names:
                out[cam] = str(video_dir / fname)
                break
    return out


def _instructions(root: Path, task_id: str, *, sources: dict | None = None) -> dict[str, str]:
    """Index per-episode annotations once per task, without borrowing another episode's text."""
    path = root / "task_info" / f"task_{task_id}.json"
    raw = path.read_bytes() if path.is_file() else None
    if sources is not None:
        sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest() if raw is not None else None
    if raw is None:
        return {}
    data = json.loads(raw)
    rows = data if isinstance(data, list) else [data]
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid Agibot annotation row in {path}")
        episode = row.get("episode_id")
        if episode is None:
            if isinstance(data, list):
                raise ValueError(f"per-episode Agibot annotation missing episode_id: {path}")
            key = "*"  # Legacy task-wide dictionary.
        else:
            key = str(int(episode))
        if key in result:
            raise ValueError(f"duplicate Agibot episode_id {key} in {path}")
        text = next((row[k] for k in ("english", "instruction", "task", "task_name")
                     if isinstance(row.get(k), str) and row[k].strip()), "")
        result[key] = text.strip()
    return result


def read_vectors(record: EpisodeRecord, spec: CustomSpec, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(record.path, "r") as f:
        t = int(f["state/joint/position"].shape[0])
        state = np.concatenate(
            [
                _h5("state/joint/position", f, t, 14),
                _h5("state/effector/position", f, t, 2),
                _h5("state/head/position", f, t, 2),
                _h5("state/waist/position", f, t, 2),
            ],
            axis=1,
        )
        action = np.concatenate(
            [
                _h5("action/joint/position", f, t, 14),
                _h5("action/effector/position", f, t, 2),
                _h5("action/head/position", f, t, 2),
                _h5("action/waist/position", f, t, 2),
                _h5("action/robot/velocity", f, t, 2),
            ],
            axis=1,
        )
    return fit_dim(state, spec.state_dim), fit_dim(action, spec.action_dim)


def _cam_video_path(video_dir: Path, cam: str) -> Path | None:
    prefix = _CAM.get(cam, cam)
    for suffix in ("_color.mp4", "_color_h264.mp4", ".mp4"):
        candidate = video_dir / f"{prefix}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_frames(record: EpisodeRecord, spec: CustomSpec, cam: str, indices: list[int]) -> np.ndarray:
    raw = (record.extra or {}).get("videos", {}).get(cam)
    path = Path(raw) if raw else _cam_video_path(Path((record.extra or {}).get("video_dir") or ""), cam)
    return read_mp4_indices(
        path or Path((record.extra or {}).get("video_dir") or "") / f"{_CAM.get(cam, cam)}.mp4",
        indices,
        fallback_hw=(spec.image_size, spec.image_size),
    )


@lru_cache(maxsize=2)
def _subtask_annotations(path: str, digest: str):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Agibot annotations changed after scanning; rescan before training')
    data = json.loads(raw)
    rows = data if isinstance(data, list) else [data]
    return {str(int(row['episode_id'])): row.get('label_info', {}).get('action_config', [])
            for row in rows if 'episode_id' in row}


def read_subtasks(record):
    """Use explicit [start_frame, end_frame) training windows; never extend unlabeled tails.

    This conservative endpoint policy excludes end_frame. It is a training
    policy, not a claim that the publisher formally specifies endpoint ownership.
    """
    sources = record.extra.get('annotation_sources', {})
    if len(sources) != 1:
        raise ValueError('Agibot subtask mode requires a freshly scanned annotation source')
    path, digest = next(iter(sources.items()))
    if digest is None:
        raise ValueError('Agibot subtask annotation file is missing')
    rows = _subtask_annotations(path, digest).get(record.extra['episode_id'], [])
    return [dict(start=row['start_frame'], stop=row['end_frame'], text=row.get('action_text', '')) for row in rows]
