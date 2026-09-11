"""LeRobot v2 (per-episode parquet/mp4) and v3 (packed files + timestamps).

Each scanned episode stores a ``LerobotDump`` in ``record.extra["lerobot"]``.
Scan, video resolve, and frame/vector reads all go through that object.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.common.lerobot_rows import (
    V2_DATA_PATH,
    V3_DATA_PATH,
    _info_field,
    cam_keys,
    chunk_from_path,
    chunks_size,
    digits,
    feature_dtype,
    info_layout,
    language,
    load_tasks,
    row_get,
    row_int,
)
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import CustomSpec
from lbm.dataloader.custom.video import contiguous_span, read_mp4_all, read_mp4_indices, read_mp4_span


@dataclass(frozen=True)
class LerobotCam:
    """One camera's mp4 location. v3 episodes share packed files via ``from_timestamp``."""

    video_key: str
    chunk: int = 0
    file_index: int = 0
    from_timestamp: float = 0.0


@dataclass
class LerobotDump:
    repo: Path
    version: str
    episode_index: int
    parquet: Path
    info: dict
    video_tmpl: str
    fps: float
    chunk: int = 0
    videos: dict[str, LerobotCam] = field(default_factory=dict)

    @property
    def is_v3(self) -> bool:
        return self.version == "v3"

    def cam_dtype(self, cam: str) -> str:
        return feature_dtype(self.info, cam)

    def resolve_mp4(self, cam: str) -> tuple[Path, str] | None:
        if self.cam_dtype(cam) == "image":
            return None
        if self.is_v3:
            if cam not in self.videos:
                return None
            meta = self.videos[cam]
            keys = [meta.video_key, *cam_keys(cam)]
            seen: set[str] = set()
            for key in keys:
                if key in seen:
                    continue
                seen.add(key)
                path = self._format_video(key, chunk=meta.chunk, episode_index=self.episode_index, file_index=meta.file_index)
                if path is not None and path.is_file():
                    return path, key
            return None
        for key in cam_keys(cam):
            path = self._format_video(key, chunk=self.chunk, episode_index=self.episode_index, file_index=0)
            if path is not None and path.is_file():
                return path, key
        return None

    def mp4_indices(self, cam: str, indices: list[int]) -> list[int]:
        if not self.is_v3:
            return [int(i) for i in indices]
        if cam not in self.videos:
            raise KeyError(f"v3 camera {cam!r} has no packed-video meta (from_timestamp)")
        meta = self.videos[cam]
        return [int(round(float(meta.from_timestamp) * self.fps)) + int(i) for i in indices]

    def _format_video(self, video_key: str, *, chunk: int, episode_index: int, file_index: int) -> Path | None:
        try:
            rel = self.video_tmpl.format(
                video_key=video_key,
                episode_chunk=chunk,
                chunk_index=chunk,
                episode_index=episode_index,
                file_index=file_index,
            )
        except (KeyError, ValueError):
            return None
        return self.repo / rel


def lerobot_of(record: EpisodeRecord) -> LerobotDump | None:
    extra = record.extra or {}
    dump = extra.get("lerobot")
    return dump if isinstance(dump, LerobotDump) else None


def record_from_dump(dump: LerobotDump, n_frames: int, lang: str = "") -> EpisodeRecord:
    """Same record scan writes: ``path`` is the episode parquet, dump lives in ``extra``."""
    return EpisodeRecord(
        kind="lerobot_v3" if dump.is_v3 else "lerobot_v2",
        path=str(dump.parquet),
        n_frames=n_frames,
        lang=lang,
        extra={"lerobot": dump},
    )


def discover_lerobot_roots(root: Path, *, max_depth: int = 5) -> Iterator[Path]:
    """DFS so ``max_episodes`` can stop before the whole tree is listed."""
    from lbm.dataloader.custom.common.fs import child_dirs

    root = Path(root)
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        if os.path.isfile(cur / "meta" / "info.json"):
            yield cur
            continue
        if depth >= max_depth:
            continue
        for child in sorted(child_dirs(cur), key=lambda p: p.name, reverse=True):
            stack.append((child, depth + 1))


def _list_lerobot_roots(root: Path, *, max_depth: int = 5) -> list[Path]:
    """Full discover: first-level subtrees in parallel."""
    from lbm.dataloader.custom.common.fs import child_dirs, map_threads

    def _walk(cur: Path, depth: int) -> list[Path]:
        if os.path.isfile(cur / "meta" / "info.json"):
            return [cur]
        if depth >= max_depth:
            return []
        kids = child_dirs(cur)
        if depth == 0 and len(kids) > 1:
            return [p for chunk in map_threads(lambda kid: _walk(kid, 1), kids) for p in chunk]
        found: list[Path] = []
        for kid in kids:
            found.extend(_walk(kid, depth + 1))
        return found

    found = _walk(Path(root), 0)
    found.sort(key=lambda p: str(p))
    return found


def parse_lerobot_version(info: dict) -> str:
    raw = str(_info_field(info, "codebase_version")).strip().lower()
    if raw.startswith("v3"):
        return "v3"
    if raw.startswith("v2"):
        return "v2"
    raise ValueError(f"unsupported codebase_version {info['codebase_version']!r}")


def scan_lerobot(root: Path, spec: CustomSpec, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    from lbm.utils.progress import track

    records: list[EpisodeRecord] = []
    remaining = max_episodes
    # Full scan lists every repo in parallel; max_episodes walks DFS and can stop early.
    repos = _list_lerobot_roots(root) if max_episodes is None else discover_lerobot_roots(root)
    for repo in track(repos, desc=f"scan {spec.name}", unit="repo", leave=False):
        info = json.loads((repo / "meta" / "info.json").read_text(encoding="utf-8"))
        version = parse_lerobot_version(info)
        scanned = (
            _scan_v3(repo, spec, info, max_episodes=remaining)
            if version == "v3"
            else _scan_v2(repo, spec, info, max_episodes=remaining)
        )
        records.extend(scanned)
        if max_episodes is not None and len(records) >= max_episodes:
            return records[:max_episodes]
        if remaining is not None:
            remaining = max_episodes - len(records)
    return records


def _scan_v2(repo: Path, spec: CustomSpec, info: dict, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    tasks = load_tasks(repo / "meta")
    data_tmpl, video_tmpl, fps = info_layout(info)
    size = chunks_size(info)
    from lbm.utils.progress import track

    records: list[EpisodeRecord] = []
    for row in track(_v2_rows(repo), desc=f"scan {repo.name}", unit="ep", leave=False):
        epi = row_int(row, "episode_index", default=0)
        n = row_int(row, "length", "num_frames", default=0)
        chunk = row_int(row, "episode_chunk", "chunk")
        if chunk is None:
            chunk = int(epi) // size
        parquet, chunk = _v2_locate_parquet(repo, data_tmpl, epi, chunk, row_get(row, "parquet"))
        if n <= 0:
            parquet, chunk, n = _v2_resolve_length(repo, parquet, chunk, epi)
        if n <= 0:
            continue
        videos = {
            cam: LerobotCam(video_key=f"observation.images.{cam}", chunk=chunk)
            for cam in spec.camera_keys
            if feature_dtype(info, cam) == "video"
        }
        records.append(
            record_from_dump(
                LerobotDump(
                    repo=repo,
                    version="v2",
                    episode_index=epi,
                    parquet=parquet,
                    info=info,
                    video_tmpl=video_tmpl,
                    fps=fps,
                    chunk=chunk,
                    videos=videos,
                ),
                n,
                language(row, tasks),
            )
        )
        if max_episodes is not None and len(records) >= max_episodes:
            break
    return records


def _v2_rows(repo: Path) -> Iterator[dict]:
    jsonl = repo / "meta" / "episodes.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    for parquet in _parquet_hits(repo / "data"):
        yield {
            "episode_index": digits(parquet.stem),
            "length": _parquet_num_rows(parquet),
            "episode_chunk": chunk_from_path(parquet),
            "parquet": str(parquet),
        }


def _parquet_hits(root: Path) -> list[Path]:
    from lbm.dataloader.custom.common.fs import list_files

    paths = list_files(root, suffixes=(".parquet",), dir_depth=2)
    return paths or list_files(root, suffixes=(".parquet",))


def _v2_locate_parquet(
    repo: Path, data_tmpl: str, epi: int, chunk: int, explicit: str | None
) -> tuple[Path, int]:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else repo / path, chunk
    try:
        path = repo / data_tmpl.format(episode_chunk=chunk, episode_index=epi, chunk_index=chunk)
    except (KeyError, ValueError):
        path = repo / V2_DATA_PATH.format(
            episode_chunk=int(chunk), episode_index=int(epi), chunk_index=int(chunk)
        )
    return path, chunk


def _v2_resolve_length(repo: Path, parquet: Path, chunk: int, epi: int) -> tuple[Path, int, int]:
    if parquet.is_file():
        return parquet, chunk, _parquet_num_rows(parquet)
    needle = f"episode_{int(epi):06d}.parquet"
    for path in _parquet_hits(repo / "data"):
        if path.name == needle:
            return path, chunk_from_path(path), _parquet_num_rows(path)
    return parquet, chunk, 0


def _scan_v3(repo: Path, spec: CustomSpec, info: dict, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    tasks = load_tasks(repo / "meta")
    data_tmpl, video_tmpl, fps = info_layout(info)
    from lbm.utils.progress import track

    ep_files = _parquet_hits(repo / "meta" / "episodes")
    ctx = (repo, spec, info, tasks, data_tmpl, video_tmpl, fps)

    def _one(path: Path) -> list[EpisodeRecord]:
        return _v3_records_from_file(path, *ctx)

    records: list[EpisodeRecord] = []
    if max_episodes is None and len(ep_files) > 1:
        from lbm.dataloader.custom.common.fs import map_threads

        records = [rec for chunk in map_threads(_one, ep_files, desc=f"scan {spec.name}") for rec in chunk]
    else:
        files = track(ep_files, desc=f"scan {spec.name}", unit="file", leave=False) if len(ep_files) > 1 else ep_files
        for ep_path in files:
            records.extend(_one(ep_path))
            if max_episodes is not None and len(records) >= max_episodes:
                return records[:max_episodes]
    if records:
        return records
    return _scan_v3_from_data(repo, info, video_tmpl, fps, max_episodes=max_episodes)


def _v3_records_from_file(
    ep_path: Path,
    repo: Path,
    spec: CustomSpec,
    info: dict,
    tasks: dict,
    data_tmpl: str,
    video_tmpl: str,
    fps: float,
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for row in pd.read_parquet(ep_path).to_dict("records"):
        n = row_int(row, "length", default=0)
        if n <= 0:
            continue
        d_chunk = row_int(row, "data/chunk_index", default=0)
        d_file = row_int(row, "data/file_index", default=0)
        try:
            parquet = repo / data_tmpl.format(chunk_index=d_chunk, file_index=d_file)
        except (KeyError, ValueError):
            parquet = repo / V3_DATA_PATH.format(chunk_index=d_chunk, file_index=d_file)
        records.append(
            record_from_dump(
                LerobotDump(
                    repo=repo,
                    version="v3",
                    episode_index=row_int(row, "episode_index", default=0),
                    parquet=parquet,
                    info=info,
                    video_tmpl=video_tmpl,
                    fps=fps,
                    videos=_v3_videos(row, spec, info, d_chunk, d_file),
                ),
                n,
                language(row, tasks),
            )
        )
    return records


def _v3_videos(row, spec: CustomSpec, info: dict, d_chunk: int, d_file: int) -> dict[str, LerobotCam]:
    out: dict[str, LerobotCam] = {}
    for cam in spec.camera_keys:
        if feature_dtype(info, cam) == "image":
            continue
        meta = _v3_cam_row(row, cam, d_chunk, d_file)
        if meta is not None:
            out[cam] = meta
    return out


def _v3_cam_row(row, cam: str, d_chunk: int, d_file: int) -> LerobotCam | None:
    for vkey in cam_keys(cam):
        ck = f"videos/{vkey}/chunk_index"
        fk = f"videos/{vkey}/file_index"
        tk = f"videos/{vkey}/from_timestamp"
        if row_get(row, ck) is None and row_get(row, fk) is None:
            continue
        return LerobotCam(
            video_key=vkey,
            chunk=row_int(row, ck, default=d_chunk),
            file_index=row_int(row, fk, default=d_file),
            from_timestamp=float(row_get(row, tk) or 0.0),
        )
    return None


def _scan_v3_from_data(
    repo: Path, info: dict, video_tmpl: str, fps: float, *, max_episodes: int | None
) -> list[EpisodeRecord]:
    from lbm.utils.progress import track

    records: list[EpisodeRecord] = []
    parquets = _parquet_hits(repo / "data")
    for parquet in track(parquets, desc=f"scan {repo.name} data", unit="file", leave=False):
        frame = pd.read_parquet(parquet, columns=["episode_index"])
        for epi, group in frame.groupby("episode_index"):
            records.append(
                record_from_dump(
                    LerobotDump(
                        repo=repo,
                        version="v3",
                        episode_index=int(epi),
                        parquet=parquet,
                        info=info,
                        video_tmpl=video_tmpl,
                        fps=fps,
                    ),
                    int(len(group)),
                )
            )
            if max_episodes is not None and len(records) >= max_episodes:
                return records
    return records


def _frame_columns(frame) -> set[str]:
    return set(frame.columns)


def read_lerobot_vectors(
    record: EpisodeRecord,
    spec: CustomSpec,
    *,
    mmap_store=None,
    action_freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = _load_table(record, mmap_store)
    state = _vector_from_frame(frame, spec.state_columns, spec.state_dim, what="state")
    have = _frame_columns(frame)
    missing = [name for name in spec.action_columns if name not in have]
    if not missing:
        action = _vector_from_frame(frame, spec.action_columns, spec.action_dim, what="action")
        return state, action
    if len(missing) != len(spec.action_columns):
        raise KeyError(f"action key(s) {missing} not found; available: {sorted(have)}")
    from lbm.action_space import derive_absolute_actions

    dump = lerobot_of(record)
    native_fps = float(dump.fps) if dump is not None and dump.fps else spec.fps
    freq = native_fps if action_freq is None else float(action_freq)
    action = derive_absolute_actions(
        state,
        spec,
        native_fps=native_fps,
        action_freq=freq,
    )
    return state, action


def resolve_lerobot_video(record: EpisodeRecord, _spec: CustomSpec, cam: str) -> tuple[Path, str, int] | None:
    dump = lerobot_of(record)
    if dump is None:
        return None
    found = dump.resolve_mp4(cam)
    if found is None:
        return None
    path, key = found
    return path, key, dump.episode_index


def read_lerobot_frames(
    record: EpisodeRecord,
    spec: CustomSpec,
    cam: str,
    indices: list[int],
    *,
    mmap_store=None,
    progress: bool = False,
) -> np.ndarray:
    dump = lerobot_of(record)
    if dump is None:
        raise KeyError("LeRobot frame read requires extra['lerobot']")
    dtype = dump.cam_dtype(cam)
    if dtype == "image":
        frames = _read_parquet_images(record, cam, indices, mmap_store, progress=progress)
        if frames is None:
            raise KeyError(f"camera {cam!r} marked image but has no parquet column")
        return frames
    if dtype == "video":
        return _read_lerobot_mp4(dump, spec, cam, indices, progress=progress)
    frames = _read_parquet_images(record, cam, indices, mmap_store, progress=progress)
    if frames is not None:
        return frames
    if dump.resolve_mp4(cam) is None:
        raise KeyError(
            f"camera {cam!r} is not in features and has neither a parquet image column nor an mp4"
        )
    return _read_lerobot_mp4(dump, spec, cam, indices, progress=progress)


def lerobot_source_jpegs(dump: LerobotDump, cam: str, n_frames: int) -> Iterator[bytes] | None:
    """Read bounded parquet batches instead of materializing an episode's images."""
    if dump.cam_dtype(cam) == "video":
        return None
    col = _parquet_cam_column(dump.parquet, cam)
    if col is None:
        return None

    def blobs():
        import pyarrow.parquet as pq

        columns = [col, "episode_index"] if dump.is_v3 else [col]
        count = 0
        with pq.ParquetFile(dump.parquet) as parquet:
            for batch in parquet.iter_batches(batch_size=32, columns=columns, use_threads=False):
                for row in batch.to_pylist():
                    if dump.is_v3 and int(row["episode_index"]) != dump.episode_index:
                        continue
                    if count >= n_frames:
                        return
                    blob = _parquet_image_bytes(row[col], repo=dump.repo)
                    if not blob:
                        raise ValueError(f"invalid image: {dump.parquet}, camera={cam}, frame={count}")
                    count += 1
                    yield blob
                if count >= n_frames:
                    return
        if count != n_frames:
            raise ValueError(f"expected {n_frames} images, got {count}: {dump.parquet}, camera={cam}")

    return blobs()


def _read_lerobot_mp4(
    dump: LerobotDump,
    spec: CustomSpec,
    cam: str,
    indices: list[int],
    *,
    progress: bool = False,
) -> np.ndarray:
    found = dump.resolve_mp4(cam)
    if found is None:
        raise KeyError(f"camera {cam!r} has no mp4 file")
    path, _key = found
    file_idx = dump.mp4_indices(cam, indices)
    span = contiguous_span(file_idx)
    if span is not None:
        frames = read_mp4_span(path, span[0], span[1], progress=progress)
        if frames is not None and frames.shape[0] == span[1] - span[0]:
            return frames
        if span[0] == 0:
            frames = read_mp4_all(path, progress=progress)
            if frames.shape[0] >= span[1]:
                return frames[: span[1]]
    return read_mp4_indices(path, file_idx, fallback_hw=(spec.image_size, spec.image_size))


def _read_parquet_images(
    record: EpisodeRecord, cam: str, indices: list[int], _mmap_store, *, progress: bool = False
) -> np.ndarray | None:
    dump = lerobot_of(record)
    if dump is None:
        raise KeyError("LeRobot image column read requires extra['lerobot']")
    col = _parquet_cam_column(dump.parquet, cam)
    if col is None:
        return None
    if dump.is_v3:
        frame = _read_episode_parquet(dump.parquet, dump.episode_index, columns=[col])
    else:
        frame = pd.read_parquet(dump.parquet, columns=[col])
    n = int(record.n_frames) if record.n_frames else 0
    cells = [_cell_at(frame, col, min(i, n - 1) if n else int(i)) for i in indices]
    if progress:
        from lbm.utils.progress import track

        cells = track(cells, desc=f"decode {cam}", total=len(cells), unit="f", leave=False)
    return np.stack([_decode_parquet_image(cell, repo=dump.repo) for cell in cells], axis=0)


def _parquet_cam_column(path: Path, cam: str) -> str | None:
    try:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return None
    for key in (cam, f"observation.images.{cam}"):
        if key in names:
            return key
    return None


def _load_table(record: EpisodeRecord, mmap_store=None):
    dump = lerobot_of(record)
    if dump is None:
        raise KeyError("LeRobot table read requires extra['lerobot']")
    parquet = dump.parquet
    if mmap_store is not None:
        if dump.is_v3:
            return mmap_store.load_episode_rows(parquet, dump.episode_index)
        return mmap_store.load(parquet)
    if dump.is_v3:
        return _read_episode_parquet(parquet, dump.episode_index)
    return pd.read_parquet(parquet)


def _cell_at(frame, col: str, index: int):
    series = frame[col]
    if hasattr(series, "iloc"):
        return series.iloc[int(index)]
    values = series.to_numpy() if hasattr(series, "to_numpy") else np.asarray(series)
    return values[int(index)]


def _parquet_image_bytes(value, *, repo: Path | str | None = None) -> bytes | None:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            return bytes(raw)
        if value.get("path"):
            path = Path(value["path"])
            if not path.is_file() and repo:
                path = Path(repo) / path
            if path.is_file():
                return path.read_bytes()
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


def _decode_parquet_image(value, *, repo: Path | str | None = None) -> np.ndarray:
    raw = _parquet_image_bytes(value, repo=repo)
    if raw is not None:
        buf = np.frombuffer(raw, dtype=np.uint8)
        try:
            import cv2
        except ImportError:
            raise KeyError("opencv required to decode parquet PNG/JPEG images") from None
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise KeyError("failed to decode parquet image bytes")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        rgb = arr[..., :3] if arr.shape[-1] == 4 else arr
        return np.asarray(rgb, dtype=np.uint8)
    raise KeyError(f"unsupported image cell type {type(value).__name__} shape={getattr(arr, 'shape', None)}")


def _read_episode_parquet(path: Path, episode_index: int, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(
            path, columns=columns, filters=[("episode_index", "=", int(episode_index))]
        )
        frame = table.to_pandas()
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["episode_index"] == int(episode_index)]
    if frame.empty:
        raise ValueError(f"episode_index={episode_index} missing in {path}")
    return frame.reset_index(drop=True)


def _parquet_num_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return int(len(pd.read_parquet(path)))


def _as_td(series) -> np.ndarray:
    values = series.to_numpy()
    if values.dtype == object:
        return np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], axis=0)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def _vector_from_frame(
    frame,
    columns: tuple[str, ...],
    dim: int,
    *,
    what: str = "vector",
) -> np.ndarray:
    have = list(frame.columns)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"{what} key(s) {missing} not found; available: {have}")
    parts = [_as_td(frame[c]) for c in columns]
    if len(parts) == 1:
        return fit_dim(parts[0], dim)
    widths = [p.shape[-1] for p in parts]
    total = sum(widths)
    if total != dim:
        raise KeyError(f"{what} columns {list(columns)} have widths {widths} (sum={total}), expected {dim}")
    return np.concatenate(parts, axis=1)
