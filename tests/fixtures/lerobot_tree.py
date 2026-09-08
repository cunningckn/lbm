"""Write a minimal LeRobot v2 tree for tests (no real videos required for init)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lbm.dataloader.custom.common.lerobot_rows import (
    V2_DATA_PATH,
    V2_VIDEO_PATH,
    V3_DATA_PATH,
    V3_VIDEO_PATH,
)
from tests.fixtures.robot_profiles import robot_io


def lerobot_info(*, version: str, fps: float, **extra) -> dict:
    """Complete ``meta/info.json`` body. Scan requires these keys."""
    v3 = str(version).startswith("v3")
    info = {
        "codebase_version": version,
        "fps": fps,
        "data_path": V3_DATA_PATH if v3 else V2_DATA_PATH,
        "video_path": V3_VIDEO_PATH if v3 else V2_VIDEO_PATH,
    }
    if not v3:
        info["chunks_size"] = 1000
    info.update(extra)
    return info


def _write_dummy_mp4(path: Path, n_frames: int, size: int = 16) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), (i * 40) % 255, dtype=np.uint8))
    writer.release()


def _slice_modality(keys: tuple[str, ...], total_dim: int, original_key: str) -> dict:
    """Evenly slice a concatenated vector across dotted keys (last key takes remainder)."""
    n = max(1, len(keys))
    base = max(1, total_dim // n)
    out: dict = {}
    start = 0
    for i, key in enumerate(keys):
        name = key.split(".", 1)[-1]
        end = total_dim if i == n - 1 else min(total_dim, start + base)
        if end <= start:
            end = start + 1
        out[name] = {"start": int(start), "end": int(end), "original_key": original_key}
        start = end
    if start < total_dim and keys:
        last = keys[-1].split(".", 1)[-1]
        out[last]["end"] = int(total_dim)
    return out


def write_lerobot_v2_tree(
    root: Path,
    *,
    robot_type: str = "rmbench",
    n_frames: int = 16,
    name: str | None = None,
    include_action: bool = True,
) -> Path:
    """Create ``root/<name>/`` with meta + one parquet episode. Returns dataset path."""
    spec = robot_io(robot_type)
    folder = root / (name or robot_type)
    meta = folder / "meta"
    data_dir = folder / "data" / "chunk-000"
    meta.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    state_dim = spec.state_dim or max(1, len(spec.state_keys))
    action_dim = spec.action_dim or max(1, len(spec.action_keys))
    fps = spec.fps or 30.0

    modality = {
        "state": _slice_modality(spec.state_keys, state_dim, "observation.state"),
        "action": _slice_modality(spec.action_keys, action_dim, "action"),
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in spec.camera_keys},
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=2))

    state_col = spec.state_keys[0] if spec.state_keys else "observation.state"
    action_col = spec.action_keys[0] if spec.action_keys else "action"

    features: dict = {
        state_col: {"dtype": "float32", "shape": [state_dim], "names": None},
    }
    if include_action:
        features[action_col] = {"dtype": "float32", "shape": [action_dim], "names": None}
    features.update({
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    })
    for cam in spec.camera_keys:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [3, 32, 32],
            "names": ["channels", "height", "width"],
            "info": {"video.height": 32, "video.width": 32, "video.channels": 3, "video.fps": fps},
        }

    info = lerobot_info(
        version="v2.1",
        fps=fps,
        robot_type=spec.embodiment,
        total_episodes=1,
        total_frames=n_frames,
        total_tasks=1,
        total_videos=len(spec.camera_keys),
        total_chunks=1,
        splits={"train": "0:1"},
        features=features,
    )
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["pick"], "length": n_frames}) + "\n")
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick"}) + "\n")

    rows = {
        "episode_index": np.zeros(n_frames, dtype=np.int64),
        "timestamp": np.arange(n_frames, dtype=np.float64) / float(fps),
        "frame_index": np.arange(n_frames, dtype=np.int64),
        "index": np.arange(n_frames, dtype=np.int64),
        "task_index": np.zeros(n_frames, dtype=np.int64),
        state_col: [np.linspace(0, 1, state_dim, dtype=np.float32) + i * 0.01 for i in range(n_frames)],
    }
    if include_action:
        rows[action_col] = [
            np.linspace(0, 1, action_dim, dtype=np.float32) * (i + 1) / n_frames for i in range(n_frames)
        ]
    pd.DataFrame(rows).to_parquet(data_dir / "episode_000000.parquet")
    for cam in spec.camera_keys:
        _write_dummy_mp4(
            folder / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{0:06d}.mp4",
            n_frames,
        )
    return folder


def write_lerobot_v3_libero_tree(root: Path, *, n_frames: int = 8) -> Path:
    """Minimal official-HF-shaped LIBERO v3 tree (image / image2, 8/7D)."""
    folder = root / "libero_hf"
    meta = folder / "meta"
    data_dir = folder / "data" / "chunk-000"
    meta.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    modality = {
        "state": {
            "state": {"start": 0, "end": 8, "original_key": "observation.state"},
        },
        "action": {
            "actions": {"start": 0, "end": 7, "original_key": "action"},
        },
        "video": {
            "image": {"original_key": "observation.images.image"},
            "image2": {"original_key": "observation.images.image2"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=2))
    features = {
        "observation.state": {"dtype": "float32", "shape": [8], "names": None},
        "action": {"dtype": "float32", "shape": [7], "names": None},
        "observation.images.image": {
            "dtype": "video",
            "shape": [3, 256, 256],
            "names": ["channels", "height", "width"],
            "info": {"video.height": 256, "video.width": 256, "video.channels": 3, "video.fps": 10},
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": [3, 256, 256],
            "names": ["channels", "height", "width"],
            "info": {"video.height": 256, "video.width": 256, "video.channels": 3, "video.fps": 10},
        },
    }
    info = lerobot_info(
        version="v3.0",
        fps=10,
        robot_type="panda",
        total_episodes=1,
        total_frames=n_frames,
        features=features,
    )
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    return folder
