"""RMBench official ``data/<task>/demo_clean`` HDF5 → one LeRobot v2.1 repo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .detect import has_info, is_lerobot, is_rmbench_hdf5
from .lerobot_v2 import write_v21

_STATE_KEYS = ("observations/qpos", "observation/qpos", "joint_action", "qpos")
_ACTION_KEYS = ("action", "joint_action", "actions")
_CAM_ALIASES = {
    "cam_high": ("cam_high", "camera_high", "head", "cam_head", "observation.images.cam_high"),
    "cam_left_wrist": (
        "cam_left_wrist",
        "cam_left",
        "camera_left",
        "left_camera",
        "observation.images.cam_left_wrist",
    ),
    "cam_right_wrist": (
        "cam_right_wrist",
        "cam_right",
        "camera_right",
        "right_camera",
        "observation.images.cam_right_wrist",
    ),
}


def convert_rmbench(src: Path, dest: Path, *, force: bool = False, fps: float = 10.0) -> Path:
    src = Path(src)
    dest = Path(dest)
    if has_info(dest) and is_lerobot(dest) and not is_rmbench_hdf5(dest) and not force:
        print(f"already LeRobot: {dest}")
        return dest
    if has_info(src) and is_lerobot(src) and not is_rmbench_hdf5(src):
        from .layout import ensure_dest

        return ensure_dest(src, dest, force=force)
    episodes_h5 = list_rmbench_h5(src)
    if not episodes_h5:
        raise FileNotFoundError(f"no data/*/demo_clean HDF5 under {src}")
    episodes = [_read_episode(path, fps=fps) for path in episodes_h5]
    dest.mkdir(parents=True, exist_ok=True)
    write_v21(dest, episodes, fps=fps, robot_type="aloha")
    print(f"rmbench: {len(episodes)} episodes -> {dest}")
    return dest


def list_rmbench_h5(root: Path) -> list[Path]:
    hits: list[Path] = []
    for pattern in (
        "data/*/demo_clean/**/*.hdf5",
        "data/*/demo_clean/**/*.h5",
        "*/demo_clean/**/*.hdf5",
        "*/demo_clean/**/*.h5",
    ):
        hits.extend(root.glob(pattern))
    uniq = sorted({p.resolve() for p in hits if p.is_file()})
    return uniq


def _read_episode(path: Path, *, fps: float) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        state = _first_array(f, _STATE_KEYS)
        action = _first_array(f, _ACTION_KEYS)
        if action is None:
            action = state
        if state is None:
            raise ValueError(f"{path}: no qpos/action")
        state = np.asarray(state, dtype=np.float32).reshape(state.shape[0], -1)
        action = np.asarray(action, dtype=np.float32).reshape(action.shape[0], -1)
        n = min(state.shape[0], action.shape[0])
        images = {}
        for cam, aliases in _CAM_ALIASES.items():
            frames = _cam_frames(f, aliases)
            if frames is not None:
                images[cam] = frames[:n]
        lang = _lang(path, f)
    return {"state": state[:n], "action": action[:n], "images": images, "lang": lang, "fps": fps}


def _first_array(f, keys: tuple[str, ...]):
    for key in keys:
        if key in f:
            return f[key][()]
    return None


def _cam_frames(f, aliases: tuple[str, ...]) -> np.ndarray | None:
    for name in aliases:
        node = _lookup(f, name)
        if node is None:
            continue
        arr = _decode_frames(node)
        if arr is not None:
            return arr
    images = _lookup(f, "observations/images")
    if images is not None:
        for name in aliases:
            if name in images:
                arr = _decode_frames(images[name])
                if arr is not None:
                    return arr
    return None


def _lookup(f, key: str):
    if key in f:
        return f[key]
    cur = f
    for part in key.split("/"):
        if part not in cur:
            return None
        cur = cur[part]
    return cur


def _decode_frames(node) -> np.ndarray | None:
    import h5py

    if isinstance(node, h5py.Dataset):
        arr = node[()]
        if arr.dtype == object or (arr.ndim == 1 and arr.dtype == np.uint8):
            return _decode_jpeg_list(arr if arr.dtype == object else None)
        arr = np.asarray(arr)
        if arr.ndim == 4:
            return arr
        return None
    if isinstance(node, h5py.Group) and "left" in node:
        return None
    return None


def _decode_jpeg_list(items) -> np.ndarray | None:
    if items is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    frames = []
    for blob in items:
        data = np.frombuffer(bytes(blob), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        frames.append(img[:, :, ::-1])
    return np.stack(frames, axis=0) if frames else None


def _lang(path: Path, f) -> str:
    if "instruction" in f.attrs:
        return str(f.attrs["instruction"])
    task = path
    for parent in path.parents:
        if parent.name == "demo_clean":
            task = parent.parent
            break
    name = task.name if task != path else path.parent.name
    for cand in (
        path.with_suffix(".json"),
        path.parent.parent / "instructions" / (path.stem + ".json"),
        path.parent.parent / "language_annotation.json",
    ):
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return str(data.get("instruction") or data.get(path.stem) or name)
            if isinstance(data, list) and data:
                return str(data[0])
    return name.replace("_", " ")
