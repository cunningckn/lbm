"""Random-access readers for raw DROID tar data and the mmap cache."""

from __future__ import annotations

import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_completion_marker, read_json, read_parquet_rows, safe_relative_path


def _scaled_intrinsics(measured: np.ndarray, height: int, width: int) -> np.ndarray:
    cx, cy = float(measured[0, 2]), float(measured[1, 2])
    if cx <= 0 or cy <= 0:
        raise ValueError(f"invalid camera principal point: {(cx, cy)}")
    affine = np.diag(
        np.array([width / (2.0 * cx), height / (2.0 * cy), 1.0], dtype=np.float32)
    )
    return (affine @ measured).astype(np.float32, copy=False)


def _selected_window(
    arrays: dict[str, np.ndarray],
    *,
    start: int,
    frames: int,
    points: int,
) -> dict[str, np.ndarray]:
    total_frames, total_points = arrays["points2d"].shape[:2]
    if total_frames <= 0 or total_points <= 0:
        raise ValueError("cannot sample an empty trajectory")
    if frames <= 0 or points <= 0:
        raise ValueError("frames and points must be positive")
    selected_frames = min(frames, total_frames)
    start = min(max(start, 0), total_frames - selected_frames)
    stop = start + selected_frames
    selected_points = min(points, total_points)
    point_indices = np.linspace(
        0, total_points - 1, num=selected_points, dtype=np.int64
    )
    result = {
        "points2d": np.ascontiguousarray(arrays["points2d"][start:stop, point_indices]),
        "points3d": np.ascontiguousarray(arrays["points3d"][start:stop, point_indices]),
        "visibility2d": np.ascontiguousarray(
            arrays["visibility2d"][start:stop, point_indices]
        ),
        "visibility3d": np.ascontiguousarray(
            arrays["visibility3d"][start:stop, point_indices]
        ),
        "intrinsics_measured": np.ascontiguousarray(arrays["intrinsics_measured"]),
        "intrinsics_ds": np.ascontiguousarray(arrays["intrinsics_ds"]),
        "extrinsics": np.ascontiguousarray(arrays["extrinsics"]),
    }
    return result


class MMapDroidReader:
    """Read portable DROID cache shards without consulting their raw source."""

    def __init__(self, cache_root: str | Path) -> None:
        self.root = Path(cache_root).resolve()
        read_completion_marker(self.root)
        self.dataset = read_json(self.root / "dataset.json")
        if self.dataset.get("format") != "molmo-motion-cache":
            raise ValueError(f"not a MolmoMotion cache: {self.root}")
        if self.dataset.get("subset") != "droid":
            raise ValueError(f"this reader only supports DROID cache data: {self.root}")
        runtime = self.dataset.get("runtime_contract", {})
        if runtime.get("all_runtime_paths_relative") is not True:
            raise ValueError("cache does not promise relative runtime paths")
        track_rows = read_parquet_rows(self.root / "tracks_index.parquet")
        camera_rows = read_parquet_rows(self.root / "cameras_index.parquet")
        clip_rows = read_parquet_rows(self.root / "clips.parquet")
        self._tracks = {row["sample_id"]: row for row in track_rows}
        self._cameras = {row["sample_id"]: row for row in camera_rows}
        self._clips = {row["sample_id"]: row for row in clip_rows}
        if len(self._tracks) != len(track_rows):
            raise ValueError("tracks_index.parquet contains duplicate sample IDs")
        if set(self._tracks) != set(self._cameras) or set(self._tracks) != set(self._clips):
            raise ValueError("track, camera, and clip index sample IDs do not agree")
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._validate_structure()

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks))

    @property
    def clips(self) -> dict[str, dict[str, Any]]:
        return self._clips

    @property
    def tracks(self) -> dict[str, dict[str, Any]]:
        return self._tracks

    def _shard_root(self, shard_relpath: str) -> Path:
        return self.root / safe_relative_path(shard_relpath)

    def _open_shard(self, shard_relpath: str) -> dict[str, np.ndarray]:
        arrays = self._arrays.get(shard_relpath)
        if arrays is not None:
            return arrays
        shard_root = self._shard_root(shard_relpath)
        arrays = {
            "points3d": np.load(shard_root / "points3d.npy", mmap_mode="r"),
            "points2d": np.load(shard_root / "points2d.npy", mmap_mode="r"),
            "visibility3d": np.load(shard_root / "visibility3d.npy", mmap_mode="r"),
            "visibility2d": np.load(shard_root / "visibility2d.npy", mmap_mode="r"),
            "camera_intrinsics_measured": np.load(
                shard_root / "camera_intrinsics_measured.npy", mmap_mode="r"
            ),
            "camera_intrinsics_ds": np.load(
                shard_root / "camera_intrinsics_ds.npy", mmap_mode="r"
            ),
            "camera_extrinsics": np.load(shard_root / "camera_extrinsics.npy", mmap_mode="r"),
        }
        self._arrays[shard_relpath] = arrays
        return arrays

    def _validate_structure(self) -> None:
        expected_offsets: dict[str, int] = defaultdict(int)
        ordered_tracks = sorted(
            self._tracks.values(), key=lambda row: (row["shard"], int(row["row_offset"]))
        )
        for track in ordered_tracks:
            shard = str(track["shard"])
            safe_relative_path(shard)
            row_offset = int(track["row_offset"])
            row_count = int(track["row_count"])
            frames = int(track["num_frames"])
            points = int(track["num_points"])
            if row_count != frames * points:
                raise ValueError(f"invalid row count for {track['sample_id']}")
            if row_offset != expected_offsets[shard]:
                raise ValueError(f"non-contiguous row offsets in shard {shard}")
            expected_offsets[shard] += row_count
        for shard, expected_rows in expected_offsets.items():
            arrays = self._open_shard(shard)
            if arrays["points3d"].shape != (expected_rows, 3):
                raise ValueError(f"points3d shape does not match index in {shard}")
            if arrays["points2d"].shape != (expected_rows, 2):
                raise ValueError(f"points2d shape does not match index in {shard}")
            if arrays["visibility3d"].shape != (expected_rows,):
                raise ValueError(f"visibility3d shape does not match index in {shard}")
            if arrays["visibility2d"].shape != (expected_rows,):
                raise ValueError(f"visibility2d shape does not match index in {shard}")
        for sample_id, camera in self._cameras.items():
            shard = str(camera["shard"])
            safe_relative_path(shard)
            arrays = self._open_shard(shard)
            camera_row = int(camera["camera_row"])
            if not 0 <= camera_row < arrays["camera_intrinsics_ds"].shape[0]:
                raise ValueError(f"camera row is out of bounds for {sample_id}")
            if arrays["camera_intrinsics_measured"].shape[1:] != (3, 3):
                raise ValueError(f"invalid measured K shape in {shard}")
            if arrays["camera_intrinsics_ds"].shape[1:] != (3, 3):
                raise ValueError(f"invalid ds K shape in {shard}")
            if arrays["camera_extrinsics"].shape[1:] != (4, 4):
                raise ValueError(f"invalid extrinsics shape in {shard}")

    def get_full(self, sample_id: str) -> dict[str, np.ndarray]:
        track = self._tracks[sample_id]
        camera = self._cameras[sample_id]
        if track["shard"] != camera["shard"]:
            raise ValueError(f"track/camera shard mismatch for {sample_id}")
        arrays = self._open_shard(str(track["shard"]))
        row_offset = int(track["row_offset"])
        row_count = int(track["row_count"])
        frames = int(track["num_frames"])
        points = int(track["num_points"])
        stop = row_offset + row_count
        camera_row = int(camera["camera_row"])
        return {
            "points3d": arrays["points3d"][row_offset:stop].reshape(frames, points, 3),
            "points2d": arrays["points2d"][row_offset:stop].reshape(frames, points, 2),
            "visibility3d": arrays["visibility3d"][row_offset:stop].reshape(frames, points),
            "visibility2d": arrays["visibility2d"][row_offset:stop].reshape(frames, points),
            "intrinsics_measured": arrays["camera_intrinsics_measured"][camera_row],
            "intrinsics_ds": arrays["camera_intrinsics_ds"][camera_row],
            "extrinsics": arrays["camera_extrinsics"][camera_row],
        }

    def get_window(
        self, sample_id: str, *, start: int = 0, frames: int = 8, points: int = 32
    ) -> dict[str, np.ndarray]:
        return _selected_window(
            self.get_full(sample_id), start=start, frames=frames, points=points
        )


class RawTarDroidReader:
    """Equivalent DROID reader that repeatedly decodes source NPZ/JSON members."""

    def __init__(self, source_root: str | Path, cache_reader: MMapDroidReader) -> None:
        self.root = Path(source_root).resolve()
        self._tracks = cache_reader.tracks
        self._clips = cache_reader.clips
        self._archives: dict[str, tarfile.TarFile] = {}
        self._camera_values: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def _archive(self, tar_relpath: str) -> tarfile.TarFile:
        safe_relative_path(tar_relpath)
        archive = self._archives.get(tar_relpath)
        if archive is None:
            archive = tarfile.open(self.root / tar_relpath, mode="r:*")
            self._archives[tar_relpath] = archive
        return archive

    def _load_npz(self, tar_relpath: str, member_name: str) -> dict[str, np.ndarray]:
        archive = self._archive(tar_relpath)
        member = archive.getmember(member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"cannot extract raw source member {member_name}")
        with extracted:
            with np.load(extracted, allow_pickle=False) as arrays:
                return {key: np.array(arrays[key], copy=True) for key in arrays.files}

    def _camera(
        self, tar_relpath: str, member_name: str, camera_id: str, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (tar_relpath, member_name, camera_id)
        cached = self._camera_values.get(cache_key)
        if cached is not None:
            return cached
        archive = self._archive(tar_relpath)
        member = archive.getmember(member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"cannot extract raw source member {member_name}")
        with extracted:
            document = json.load(extracted)
        calibration = document[camera_id]
        measured = np.asarray(calibration["measured_intrinsics"], dtype=np.float32)
        if "vggt_extrinsics" in calibration:
            extrinsics = np.asarray(calibration["vggt_extrinsics"], dtype=np.float32)
        else:
            extrinsics = np.asarray(calibration["optimized_extrinsics"], dtype=np.float32)
        if measured.shape != (3, 3) or extrinsics.shape != (4, 4):
            raise ValueError(f"invalid raw calibration shape for {camera_id}")
        self._camera_values[cache_key] = (measured, extrinsics)
        return measured, extrinsics

    def get_full(self, sample_id: str) -> dict[str, np.ndarray]:
        track = self._tracks[sample_id]
        clip = self._clips[sample_id]
        data_2d = self._load_npz(
            clip["source_track_2d_tar"], clip["source_track_2d_member"]
        )
        data_3d = self._load_npz(
            clip["source_track_3d_tar"], clip["source_track_3d_member"]
        )
        points2d = np.asarray(data_2d["tracks_2d"])
        points3d = np.asarray(data_3d["points_3d"])
        visibility2d = np.asarray(data_2d["visibility"], dtype=np.bool_)
        visibility3d = np.asarray(data_3d["valid_3d"], dtype=np.bool_)
        measured, extrinsics = self._camera(
            clip["source_camera_tar"],
            clip["source_camera_member"],
            clip["source_camera_id"],
            int(clip["height"]),
            int(clip["width"]),
        )
        if points2d.shape[:2] != (
            int(track["num_frames"]),
            int(track["num_points"]),
        ):
            raise ValueError(f"raw source shape changed for {sample_id}")
        return {
            "points3d": points3d,
            "points2d": points2d,
            "visibility3d": visibility3d,
            "visibility2d": visibility2d,
            "intrinsics_measured": measured,
            "intrinsics_ds": _scaled_intrinsics(
                measured, int(clip["height"]), int(clip["width"])
            ),
            "extrinsics": extrinsics,
        }

    def get_window(
        self, sample_id: str, *, start: int = 0, frames: int = 8, points: int = 32
    ) -> dict[str, np.ndarray]:
        return _selected_window(
            self.get_full(sample_id), start=start, frames=frames, points=points
        )
