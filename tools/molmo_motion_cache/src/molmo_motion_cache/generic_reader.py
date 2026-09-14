"""Portable random-access reader for materialized non-DROID subsets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_json, read_parquet_rows, safe_relative_path
from .generic import GENERIC_SUBSETS


TRACK_ARRAYS = ("points2d", "points3d", "visibility2d", "visibility3d")
CAMERA_ARRAYS = (
    "camera_poses",
    "camera_pose_indices",
    "camera_intrinsics_dynamic",
    "camera_intrinsic_indices",
    "camera_intrinsics_static",
)
EMPTY_CAMERA_ARRAYS = {
    "camera_poses": ((0, 4, 4), np.float32),
    "camera_pose_indices": ((0,), np.int64),
    "camera_intrinsics_dynamic": ((0, 4), np.float32),
    "camera_intrinsic_indices": ((0,), np.int64),
    "camera_intrinsics_static": ((0, 3, 3), np.float32),
}


class MMapMotionReader:
    """Read one generic subset without consulting raw tar, NPZ, or JSON files."""

    def __init__(self, cache_root: str | Path) -> None:
        self.root = Path(cache_root).resolve()
        self.dataset = read_json(self.root / "dataset.json")
        if self.dataset.get("format") != "molmo-motion-cache":
            raise ValueError(f"not a MolmoMotion cache: {self.root}")
        self.subset = str(self.dataset.get("dataset", ""))
        if self.subset not in GENERIC_SUBSETS:
            raise ValueError(f"not a materialized generic subset: {self.root}")
        runtime = self.dataset.get("runtime_contract", {})
        if runtime.get("all_runtime_paths_relative") is not True:
            raise ValueError("cache does not promise relative runtime paths")

        clip_rows = read_parquet_rows(self.root / "clips.parquet")
        track_rows = read_parquet_rows(self.root / "tracks_index.parquet")
        camera_rows = read_parquet_rows(self.root / "cameras_index.parquet")
        self.clips = {str(row["sample_id"]): row for row in clip_rows}
        self.tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in track_rows:
            self.tracks[str(row["sample_id"])].append(row)
        self.cameras = {str(row["sample_id"]): row for row in camera_rows}
        if len(self.clips) != len(clip_rows) or len(self.cameras) != len(camera_rows):
            raise ValueError("clip or camera index contains duplicate sample IDs")
        if set(self.clips) != set(self.tracks) or set(self.clips) != set(self.cameras):
            raise ValueError("clip, track, and camera sample IDs do not agree")
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        for rows in self.tracks.values():
            seen: set[str] = set()
            for row in rows:
                object_id = str(row["object_id"])
                if object_id in seen:
                    raise ValueError(f"duplicate object ID for {row['sample_id']}: {object_id}")
                seen.add(object_id)
                safe_relative_path(str(row["shard"]))
                if int(row["row_count"]) != int(row["num_frames"]) * int(row["num_points"]):
                    raise ValueError(f"invalid row count for {row['sample_id']}/{object_id}")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.clips))

    def object_ids(self, sample_id: str) -> tuple[str, ...]:
        return tuple(str(row["object_id"]) for row in self.tracks[sample_id])

    def _array(self, shard: str, name: str, *, required: bool = True) -> np.ndarray:
        relative = safe_relative_path(shard)
        key = (shard, name)
        cached = self._arrays.get(key)
        if cached is not None:
            return cached
        path = self.root / relative / f"{name}.npy"
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            shape, dtype = EMPTY_CAMERA_ARRAYS[name]
            return np.empty(shape, dtype=dtype)
        array = np.load(path, mmap_mode="r")
        self._arrays[key] = array
        return array

    def _track_row(self, sample_id: str, object_id: str | None) -> dict[str, Any]:
        rows = self.tracks[sample_id]
        if object_id is None:
            return rows[0]
        for row in rows:
            if str(row["object_id"]) == object_id:
                return row
        raise KeyError(f"unknown object {object_id!r} for {sample_id}")

    def get_object_full(
        self, sample_id: str, object_id: str | None = None
    ) -> dict[str, np.ndarray]:
        row = self._track_row(sample_id, object_id)
        shard = str(row["shard"])
        start = int(row["row_offset"])
        count = int(row["row_count"])
        frames = int(row["num_frames"])
        points = int(row["num_points"])
        stop = start + count
        camera = self.cameras[sample_id]
        if str(camera["shard"]) != shard:
            raise ValueError(f"track/camera shard mismatch for {sample_id}")
        result = {
            "points2d": self._array(shard, "points2d")[start:stop].reshape(frames, points, 2),
            "points3d": self._array(shard, "points3d")[start:stop].reshape(frames, points, 3),
            "visibility2d": self._array(shard, "visibility2d")[start:stop].reshape(frames, points),
            "visibility3d": self._array(shard, "visibility3d")[start:stop].reshape(frames, points),
        }
        trust_path = self.root / safe_relative_path(shard) / "trust_weights.npy"
        if bool(row.get("trust_weights_available")):
            if not trust_path.is_file():
                raise FileNotFoundError(trust_path)
            result["trust_weights"] = self._array(shard, "trust_weights")[start:stop].reshape(
                frames, points
            )
        keep_mask = row.get("keep_mask")
        if keep_mask is not None:
            result["keep_mask"] = np.asarray(keep_mask, dtype=np.bool_)
        camera_slices = (
            ("camera_poses", "pose_offset", "pose_count"),
            ("camera_pose_indices", "pose_offset", "pose_count"),
            (
                "camera_intrinsics_dynamic",
                "dynamic_intrinsic_offset",
                "dynamic_intrinsic_count",
            ),
            (
                "camera_intrinsic_indices",
                "dynamic_intrinsic_offset",
                "dynamic_intrinsic_count",
            ),
            (
                "camera_intrinsics_static",
                "static_intrinsic_offset",
                "static_intrinsic_count",
            ),
        )
        for name, offset_key, count_key in camera_slices:
            amount = int(camera[count_key])
            offset = int(camera[offset_key])
            result[name] = self._array(shard, name, required=amount > 0)[offset : offset + amount]
        return result

    def get_object_window(
        self,
        sample_id: str,
        object_id: str | None = None,
        *,
        start: int = 0,
        frames: int = 8,
        points: int = 32,
    ) -> dict[str, np.ndarray]:
        if frames <= 0 or points <= 0:
            raise ValueError("frames and points must be positive")
        full = self.get_object_full(sample_id, object_id)
        total_frames, total_points = full["points2d"].shape[:2]
        selected_frames = min(frames, total_frames)
        start = min(max(start, 0), total_frames - selected_frames)
        stop = start + selected_frames
        point_indices = np.linspace(
            0, total_points - 1, num=min(points, total_points), dtype=np.int64
        )
        result = {
            name: np.ascontiguousarray(full[name][start:stop, point_indices])
            for name in TRACK_ARRAYS
        }
        if "trust_weights" in full:
            result["trust_weights"] = np.ascontiguousarray(
                full["trust_weights"][start:stop, point_indices]
            )
        if "keep_mask" in full:
            result["keep_mask"] = np.ascontiguousarray(full["keep_mask"])
        for name in CAMERA_ARRAYS:
            result[name] = np.ascontiguousarray(full[name])
        return result
