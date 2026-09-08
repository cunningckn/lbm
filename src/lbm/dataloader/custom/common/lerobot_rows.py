"""Parse LeRobot jsonl / parquet episode rows (dict or pandas Series)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Official LeRobot info.json templates. Scan reads them from the dump; tests reuse these.
V2_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V2_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def cam_keys(cam: str) -> tuple[str, ...]:
    return (f"observation.images.{cam}", cam)


def _info_field(info: dict, key: str):
    if key not in info:
        raise KeyError(f"meta/info.json missing {key!r}")
    value = info[key]
    if value is None or value == "":
        raise KeyError(f"meta/info.json {key!r} is empty")
    return value


def info_layout(info: dict) -> tuple[str, str, float]:
    """``data_path``, ``video_path``, ``fps`` from ``meta/info.json``. Missing keys raise."""
    return (
        str(_info_field(info, "data_path")),
        str(_info_field(info, "video_path")),
        float(_info_field(info, "fps")),
    )


def as_int(value, default: int | None = 0) -> int | None:
    if value is None:
        return default
    try:
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return default
    except TypeError:
        pass
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def row_get(row, key, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    index = getattr(row, "index", None)
    if index is not None:
        return row[key] if key in index else default
    return default


def row_int(row, *keys, default: int | None = None) -> int | None:
    for key in keys:
        parsed = as_int(row_get(row, key), default=None)
        if parsed is not None:
            return parsed
    return default


def chunks_size(info: dict) -> int:
    size = as_int(_info_field(info, "chunks_size"), default=None)
    if size is None or size < 1:
        raise KeyError(f"meta/info.json chunks_size must be a positive int, got {info['chunks_size']!r}")
    return size


def feature_dtype(info: dict, cam: str) -> str:
    features = info.get("features") or {}
    for key in cam_keys(cam):
        feat = features.get(key)
        if isinstance(feat, dict) and feat.get("dtype"):
            return str(feat["dtype"])
    return ""


def digits(stem: str) -> int:
    chars = "".join(ch for ch in stem if ch.isdigit())
    return int(chars) if chars else 0


def chunk_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("chunk-"):
            try:
                return int(part.split("-", 1)[1])
            except ValueError:
                return 0
    return 0


def load_tasks(meta: Path) -> dict[int, str]:
    jsonl = meta / "tasks.jsonl"
    if jsonl.is_file():
        out: dict[int, str] = {}
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row.get("task_index", len(out)))] = str(row.get("task") or "")
        return out
    parquet = meta / "tasks.parquet"
    if parquet.is_file():
        frame = pd.read_parquet(parquet)
        col = "task" if "task" in frame.columns else frame.columns[-1]
        idx = frame["task_index"] if "task_index" in frame.columns else range(len(frame))
        return {int(i): str(t) for i, t in zip(idx, frame[col], strict=False)}
    return {}


def language(row, tasks: dict[int, str]) -> str:
    cell = row_get(row, "tasks")
    if isinstance(cell, str) and cell.strip():
        return cell
    if isinstance(cell, np.ndarray):
        cell = cell.reshape(-1).tolist()
    if isinstance(cell, list) and cell:
        val = cell[0]
        while isinstance(val, (list, tuple, np.ndarray)) and len(val):
            val = val[0] if not isinstance(val, np.ndarray) else val.reshape(-1)[0]
        return str(val)
    idx = row_int(row, "task_index")
    if idx is not None and tasks:
        return str(tasks.get(idx) or next(iter(tasks.values()), ""))
    return str(tasks.get(0, "") or next(iter(tasks.values()), ""))
