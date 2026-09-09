"""Official DAS HDF5 (cameras inside) → slim episode.hdf5 + wrist mp4 + meta."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .detect import is_das_official_hdf5, is_das_slim
from .layout import ensure_dest
from .lerobot_v2 import write_mp4


def convert_das(src: Path, dest: Path, *, force: bool = False) -> Path:
    src = Path(src)
    dest = Path(dest)
    if is_das_slim(dest) and not force:
        print(f"already slim: {dest}")
        return dest
    if is_das_slim(src):
        return ensure_dest(src, dest, force=force)
    if not is_das_official_hdf5(src):
        raise FileNotFoundError(
            f"{src} is not DAS slim and not official HDF5 (observations/eef_pos). "
            "Full 10Kh MCAP is not converted here: https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData"
        )
    dest.mkdir(parents=True, exist_ok=True)
    task = dest / "sample"
    episodes: list[str] = []
    for i, h5_path in enumerate(_iter_h5(src)):
        rel = f"{i:05d}/episode.hdf5"
        out_dir = task / f"{i:05d}"
        _convert_one(h5_path, out_dir)
        episodes.append(rel)
        print(f"das: {h5_path.name} -> {out_dir}")
    meta = {"episodes": episodes, "source": str(src)}
    (task / "das_gripper_slim_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest


def _iter_h5(root: Path) -> list[Path]:
    paths = [p for p in root.rglob("*") if p.suffix.lower() in {".h5", ".hdf5"} and p.is_file()]
    return sorted(paths)


def _convert_one(src: Path, out_dir: Path) -> None:
    import h5py

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f:
        eef = np.asarray(f["observations/eef_pos"][()], dtype=np.float64)
        if eef.ndim == 1:
            eef = eef[None, :]
        t = eef.shape[0]
        pose = eef[:, :7] if eef.shape[1] >= 7 else np.pad(eef, ((0, 0), (0, 7 - eef.shape[1])))
        mag = eef[:, 7] if eef.shape[1] >= 8 else np.zeros(t, dtype=np.float64)
        cameras = _cameras(f)
        left_frames = _pick_cam(cameras, ("left", "wrist_left", "cam_left"))
        right_frames = _pick_cam(cameras, ("right", "wrist_right", "cam_right"))
        if left_frames is None and cameras:
            left_frames = next(iter(cameras.values()))
        if right_frames is None:
            names = [n for n in cameras if n != (_name_of(left_frames, cameras) if left_frames is not None else "")]
            if names:
                right_frames = cameras[names[0]]
        with h5py.File(out_dir / "episode.hdf5", "w") as out:
            out.create_dataset("timestamp", data=np.arange(t, dtype=np.int64))
            out.create_dataset("img_idx_for_eef", data=np.stack([np.arange(t), np.arange(t)], axis=1))
            grp = out.create_group("observations")
            grp.create_dataset("left_eef_pose", data=pose)
            grp.create_dataset("right_eef_pose", data=np.zeros_like(pose))
            grp.create_dataset("mag_left", data=mag)
            grp.create_dataset("mag_right", data=np.zeros_like(mag))
            joint = out.create_group("state").create_group("joint")
            joint.create_dataset("position", data=np.zeros((t, 14), dtype=np.float32))
        if left_frames is not None:
            write_mp4(out_dir / "cam_left_wrist.mp4", _as_uint8(left_frames[:t]), 30.0)
        if right_frames is not None:
            write_mp4(out_dir / "cam_right_wrist.mp4", _as_uint8(right_frames[:t]), 30.0)


def _cameras(f) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "observations/cameras" not in f:
        return out
    group = f["observations/cameras"]
    for name in group.keys():
        arr = np.asarray(group[name][()])
        if arr.ndim == 4:
            out[str(name)] = arr
    return out


def _pick_cam(cameras: dict[str, np.ndarray], needles: tuple[str, ...]) -> np.ndarray | None:
    lower = {k.lower(): k for k in cameras}
    for needle in needles:
        for key, orig in lower.items():
            if needle in key:
                return cameras[orig]
    return None


def _name_of(frames: np.ndarray, cameras: dict[str, np.ndarray]) -> str:
    for name, arr in cameras.items():
        if arr is frames:
            return name
    return ""


def _as_uint8(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr
