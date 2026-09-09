"""Write a LeRobot v2.1 repo (parquet + mp4 + meta) without the lerobot package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(f"expected T,H,W,C frames, got {frames.shape}")
    h, w = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    try:
        for frame in frames:
            rgb = frame[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
    finally:
        writer.release()


def write_v21(
    dest: Path,
    episodes: list[dict],
    *,
    fps: float,
    robot_type: str = "aloha",
    state_dim: int = 14,
    action_dim: int = 14,
    cams: tuple[str, ...] = CAMS,
) -> Path:
    dest = Path(dest)
    meta = dest / "meta"
    data_dir = dest / "data" / "chunk-000"
    meta.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, int] = {}
    ep_lines: list[str] = []
    total_frames = 0
    for i, ep in enumerate(episodes):
        lang = str(ep.get("lang") or "task")
        if lang not in tasks:
            tasks[lang] = len(tasks)
        state = np.asarray(ep["state"], dtype=np.float32)
        action = np.asarray(ep["action"], dtype=np.float32)
        n = int(state.shape[0])
        total_frames += n
        rows = {
            "observation.state": [state[t] for t in range(n)],
            "action": [action[t] for t in range(n)],
            "timestamp": np.arange(n, dtype=np.float32) / float(fps),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, i, dtype=np.int64),
            "index": np.arange(total_frames - n, total_frames, dtype=np.int64),
            "task_index": np.full(n, tasks[lang], dtype=np.int64),
        }
        pd.DataFrame(rows).to_parquet(data_dir / f"episode_{i:06d}.parquet")
        images = ep.get("images") or {}
        for cam in cams:
            frames = images.get(cam)
            if frames is None:
                continue
            write_mp4(
                dest / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{i:06d}.mp4",
                np.asarray(frames),
                fps,
            )
        ep_lines.append(json.dumps({"episode_index": i, "tasks": [lang], "length": n}))
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam in cams:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
        }
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (meta / "episodes.jsonl").write_text("\n".join(ep_lines) + ("\n" if ep_lines else ""), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": i, "task": t}) for t, i in tasks.items()) + ("\n" if tasks else ""),
        encoding="utf-8",
    )
    return dest
