"""Hy Embodied Lance tables (JPEG stills + 2-D gripper action)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lbm.action_space import dual_eef
from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.common.jpeg import still_rgb
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import WRISTS, CustomSpec, make_spec

NAME = "hy_lance"
SPEC = make_spec("hy_lance", "hy_lance", WRISTS, 16, 16, 30.0, 14, kind="lance", action_space=dual_eef(format="xyz_quat"))

_CAM = {
    "cam_high": "observation_images_cam_high",
    "cam_left_wrist": "observation_images_cam_left_wrist",
    "cam_right_wrist": "observation_images_cam_right_wrist",
}


def scan(root: Path, spec: CustomSpec, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    from lbm.dataloader.custom.common.fs import child_dirs

    records: list[EpisodeRecord] = []
    root = Path(root)
    tables = sorted((p for p in child_dirs(root) if p.name.startswith("table_")), key=lambda p: p.name)
    if root.is_dir() and root.name.startswith("table_"):
        tables = [root] + [p for p in tables if p.resolve() != root.resolve()]
    from lbm.utils.progress import track

    for table_dir in track(tables, desc=f"scan {spec.name}", unit="table", leave=False):
        lance_path = table_dir / f"{table_dir.name}.lance"
        if not lance_path.is_dir():
            continue
        lang_map = _task_map(table_dir)
        for epi, start, length, task_i in _episode_rows(table_dir, max_episodes=max_episodes):
            records.append(
                EpisodeRecord(
                    kind="lance",
                    path=str(lance_path),
                    n_frames=int(length),
                    lang=lang_map.get(int(task_i), ""),
                    extra={"start": int(start), "episode_index": int(epi)},
                )
            )
            if max_episodes is not None and len(records) >= max_episodes:
                return records
    return records


def _task_map(table_dir: Path) -> dict[int, str]:
    parquet = table_dir / "meta" / "tasks.parquet"
    if not parquet.is_file():
        return {}
    import pandas as pd

    frame = pd.read_parquet(parquet)
    col = "task_en" if "task_en" in frame.columns else "task" if "task" in frame.columns else None
    if col is None:
        return {}
    idx = frame["task_index"] if "task_index" in frame.columns else range(len(frame))
    return {int(i): str(t) for i, t in zip(idx, frame[col], strict=False) if t}


def _episode_rows(table_dir: Path, *, max_episodes: int | None) -> list[tuple[int, int, int, int]]:
    hypo = table_dir / "meta" / "hy_episodes.jsonl"
    if hypo.is_file():
        text = hypo.read_text(encoding="utf-8").strip()
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            row = json.loads(text.splitlines()[0])
        episodes = row.get("episodes") if isinstance(row, dict) else row
        out = []
        for item in episodes or []:
            epi, start, length = item[:3]
            task_i = int(item[3]) if len(item) > 3 else 0
            out.append((int(epi), int(start), int(length), task_i))
            if max_episodes is not None and len(out) >= max_episodes:
                break
        return out
    ep_dir = table_dir / "meta" / "episodes"
    if ep_dir.is_dir():
        import pandas as pd

        out = []
        for parquet in sorted(ep_dir.rglob("*.parquet")):
            frame = pd.read_parquet(parquet)
            for row in frame.to_dict(orient="records"):
                epi = int(row.get("episode_index") or 0)
                start = int(row.get("dataset_from_index") or row.get("from_index") or 0)
                if row.get("length") is not None:
                    length = int(row["length"])
                elif row.get("dataset_to_index") is not None:
                    length = int(row["dataset_to_index"]) - start
                else:
                    continue
                if length <= 0:
                    continue
                task_i = int(row.get("task_index") or 0)
                out.append((epi, start, length, task_i))
                if max_episodes is not None and len(out) >= max_episodes:
                    return out
        return out
    return [(0, 0, 1, 0)]


def _arrow_col(batch, name: str) -> np.ndarray:
    try:
        col = batch[name]
    except (KeyError, TypeError, ValueError):
        col = batch.column(name)
    if hasattr(col, "to_pylist"):
        values = col.to_pylist()
        return np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], axis=0)
    return np.asarray(col, dtype=np.float32)


def _names(ds) -> set[str]:
    schema = ds.schema
    if hasattr(schema, "names"):
        return set(schema.names)
    return {field.name for field in schema}


def _take(ds, indices: np.ndarray, columns: list[str]):
    import pyarrow as pa

    present = [c for c in columns if c in _names(ds)]
    if not present:
        return None
    idx = pa.array(np.asarray(indices, dtype=np.int64), type=pa.int64())
    return ds.take(idx, columns=present)


def read_vectors(record: EpisodeRecord, spec: CustomSpec, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
    import lance

    ds = lance.dataset(record.path)
    start = int(record.extra.get("start") or 0)
    n = int(record.n_frames)
    idx = np.arange(start, start + n, dtype=np.int64)
    cols = _take(ds, idx, ["observation_state", "action"])
    if cols is None:
        raise KeyError(f"lance table missing observation_state/action in {record.path}")
    state = _arrow_col(cols, "observation_state")
    action = _arrow_col(cols, "action")
    state = fit_dim(state, spec.state_dim)
    if action.ndim == 1:
        action = action[:, None]
    if action.shape[-1] == 2:
        target = np.concatenate([state[1:], state[-1:]], axis=0)
        target[:, 7] = action[:, 0]
        target[:, 15] = action[:, 1]
        action = target
    return state, fit_dim(action, spec.action_dim)


def read_frames(record: EpisodeRecord, spec: CustomSpec, cam: str, indices: list[int]) -> np.ndarray:
    import lance

    ds = lance.dataset(record.path)
    start = int(record.extra.get("start") or 0)
    col = _CAM.get(cam)
    hw = (spec.image_size, spec.image_size)
    blank = np.zeros((len(indices),) + hw + (3,), dtype=np.uint8)
    if not col:
        return blank
    idx = np.asarray([start + int(i) for i in indices], dtype=np.int64)
    batch = _take(ds, idx, [col])
    if batch is None:
        return blank
    try:
        cells = batch[col].to_pylist() if hasattr(batch[col], "to_pylist") else list(batch[col])
    except Exception:
        return blank
    imgs = [still_rgb(raw, hw) for raw in cells]
    return np.stack(imgs, axis=0) if imgs else blank
