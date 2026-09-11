"""EgoVerse / Aria zarr (JPEG ``images.front_1`` on cam_high; wrists are black + mask)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lbm.action_space import dual_eef
from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.common.jpeg import jpeg_bytes
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import WRISTS, CustomSpec, make_spec

NAME = "egoverse"
SPEC = make_spec(
    "egoverse", "egoverse", WRISTS, 16, 16, 30.0, 12, kind="zarr", action_space=dual_eef(format="xyz_quat")
)


def scan(root: Path, spec: CustomSpec, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    from lbm.dataloader.custom.common.fs import list_dirs, map_threads

    paths = list_dirs(Path(root), suffix=".zarr", max_depth=2, max_dirs=max_episodes)
    metas = map_threads(_meta, paths, desc=f"scan {spec.name}")
    records: list[EpisodeRecord] = []
    for zpath, (n, lang) in zip(paths, metas, strict=True):
        if n <= 0:
            continue
        records.append(
            EpisodeRecord(
                kind="zarr",
                path=str(zpath),
                n_frames=n,
                lang=lang,
                extra={"video_keys": {"cam_high": "images.front_1"}},
            )
        )
    return records


def _array(root: Path, name: str):
    import zarr

    direct = Path(root) / name
    if (direct / "zarr.json").is_file() or (direct / ".zarray").is_file():
        return zarr.open_array(str(direct), mode="r")
    group = zarr.open(str(root), mode="r")
    return group[name]


def _meta(path: Path) -> tuple[int, str]:
    info = path / "zarr.json"
    lang = ""
    n = 0
    if info.is_file():
        try:
            data = json.loads(info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        attrs = data.get("attributes") if isinstance(data.get("attributes"), dict) else {}
        lang = str(
            attrs.get("task_description")
            or attrs.get("task_name")
            or data.get("task_description")
            or data.get("task_name")
            or ""
        )
        n = int(attrs.get("total_frames") or 0)
    if n <= 0:
        try:
            n = int(_array(path, "left.obs_ee_pose").shape[0])
        except Exception:
            n = 0
    return n, lang


def read_vectors(
    record: EpisodeRecord,
    spec: CustomSpec,
    *,
    action_freq: float | None = None,
    **_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(_array(Path(record.path), "left.obs_ee_pose")[...], dtype=np.float32)
    right = np.asarray(_array(Path(record.path), "right.obs_ee_pose")[...], dtype=np.float32)
    grip = np.full((left.shape[0], 2), 0.5, dtype=np.float32)
    state = np.concatenate([left, grip[:, :1], right, grip[:, 1:]], axis=1)
    state = fit_dim(state, spec.state_dim)
    from lbm.action_space import derive_absolute_actions

    freq = spec.fps if action_freq is None else float(action_freq)
    action = derive_absolute_actions(state, spec, native_fps=spec.fps, action_freq=freq)
    return state, action


def _owns_front(record: EpisodeRecord, spec: CustomSpec, cam: str) -> bool:
    extra = record.extra or {}
    keys = extra.get("video_keys")
    if isinstance(keys, dict):
        return cam in keys
    if extra.get("video_key") is not None:
        return cam == spec.camera_keys[0]
    return cam == spec.camera_keys[0]


def read_frames(record: EpisodeRecord, spec: CustomSpec, cam: str, indices: list[int]) -> np.ndarray:
    import cv2

    hw = (spec.image_size, spec.image_size)
    blank = np.zeros((len(indices),) + hw + (3,), dtype=np.uint8)
    if not _owns_front(record, spec, cam):
        return blank
    try:
        stream = _array(Path(record.path), "images.front_1")
    except Exception:
        return blank
    from lbm.dataloader.custom.video import contiguous_span

    n = int(stream.shape[0])
    idx = [int(np.clip(i, 0, max(n - 1, 0))) for i in indices]
    span = contiguous_span(idx)
    cells = stream[span[0] : span[1]] if span is not None else [stream[i] for i in idx]
    frames = []
    for cell in cells:
        blob = jpeg_bytes(cell)
        img = None
        if blob:
            bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if img is None:
            img = np.zeros(hw + (3,), dtype=np.uint8)
        frames.append(img)
    return np.stack(frames, axis=0)


def mmap_source_jpegs(record: EpisodeRecord, spec: CustomSpec, cam: str):
    if not _owns_front(record, spec, cam):
        return None
    try:
        stream = _array(Path(record.path), "images.front_1")
    except Exception:
        return None
    n = int(record.n_frames) if record.n_frames else int(stream.shape[0])
    n = min(n, int(stream.shape[0]))
    if n <= 0:
        return None

    def blobs():
        for i in range(n):
            blob = jpeg_bytes(stream[i])
            if not blob:
                raise ValueError(f"invalid image: {record.path}, camera={cam}, frame={i}")
            yield blob

    return blobs()
