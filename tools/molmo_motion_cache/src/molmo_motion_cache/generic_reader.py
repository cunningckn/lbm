"""Portable random-access reader for materialized non-DROID subsets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_json, read_parquet_rows, require_ready_cache, safe_relative_path
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
        require_ready_cache(Path(cache_root))
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
        self._rgb_readers = {}
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

    def get_object_full(self, sample_id: str, object_id: str | None = None) -> dict[str, np.ndarray]:
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
            result["trust_weights"] = self._array(shard, "trust_weights")[start:stop].reshape(frames, points)
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
        point_indices = np.linspace(0, total_points - 1, num=min(points, total_points), dtype=np.int64)
        result = {name: np.ascontiguousarray(full[name][start:stop, point_indices]) for name in TRACK_ARRAYS}
        if "trust_weights" in full:
            result["trust_weights"] = np.ascontiguousarray(full["trust_weights"][start:stop, point_indices])
        if "keep_mask" in full:
            result["keep_mask"] = np.ascontiguousarray(full["keep_mask"])
        for name in CAMERA_ARRAYS:
            result[name] = np.ascontiguousarray(full[name])
        return result

    def get_selection(
        self, sample_id, *, frame_ids, point_ids, object_id=None, rgb_root=None, jpeg224_root=None, require_rgb=False
    ):
        """Explicit selection; distinguish annotation time from encoded video PTS."""
        full = self.get_object_full(sample_id, object_id)
        frames, points = np.asarray(frame_ids), np.asarray(point_ids)
        for ids, size in ((frames, full["points2d"].shape[0]), (points, full["points2d"].shape[1])):
            if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= size):
                raise IndexError("explicit frame/point index is empty, noninteger or out of range")
        result = {name: np.ascontiguousarray(full[name][frames[:, None], points]) for name in TRACK_ARRAYS}
        result.update(
            frame_ids=frames,
            point_ids=points,
            state="annotations-only",
            annotation_time_seconds=frames / float(self.clips[sample_id]["fps"]),
        )
        if rgb_root is not None and jpeg224_root is not None:
            raise ValueError("choose one RGB cache")
        if jpeg224_root is not None:
            from .geometry224 import content_mask, select_camera, transform_points
            from .rgb224 import JPEG224Reader

            if self.subset != "molmospaces":
                raise ValueError("JPEG224 source geometry is only supported for MolmoSpaces")
            key = ("jpeg224", str(Path(jpeg224_root).resolve()))
            if key not in self._rgb_readers:
                self._rgb_readers[key] = JPEG224Reader(jpeg224_root)
            reader = self._rgb_readers[key]
            clip = self.clips[sample_id]
            _, video = reader.videos[clip["video_id"]]
            geometry = video["geometry"]
            if video["count"] != full["points2d"].shape[0] or geometry["source_size"] != [
                clip["width"],
                clip["height"],
            ]:
                raise ValueError("JPEG224 source geometry/frame count mismatch")
            poses, k = select_camera(full, frames)
            p224 = transform_points(result["points2d"], geometry)
            w, h = geometry["source_size"]
            p = result["points2d"]
            bound = -0.5 if geometry["coordinate_convention"] == "integer-center" else 0
            valid = (
                np.isfinite(p).all(axis=-1)
                & (p[..., 0] >= bound)
                & (p[..., 0] < w + bound)
                & (p[..., 1] >= bound)
                & (p[..., 1] < h + bound)
                & result["visibility2d"]
            )
            result.update(reader.get(clip["video_id"], frames))
            result.update(
                points2d_source=result["points2d"],
                points2d_224=p224,
                intrinsics_source_pixels=k,
                intrinsics_224=np.asarray(geometry["A"]) @ k,
                camera_poses_source=poses,
                camera_frame_ids=frames.copy(),
                pose_convention=self.cameras[sample_id]["pose_convention"],
                content_mask=content_mask(geometry),
                supervision_valid_224=valid,
                state="jpeg224-frame-index-pilot",
            )
        elif rgb_root is not None:
            from .rgb_pilot import RGBReader

            video_id = self.clips[sample_id]["video_id"]
            key = ("legacy", str(Path(rgb_root).resolve()), video_id)
            if key not in self._rgb_readers:
                self._rgb_readers[key] = RGBReader(Path(rgb_root) / safe_relative_path(video_id))
            rgb = self._rgb_readers[key]
            if rgb.metadata["frame_count"] != full["points2d"].shape[0]:
                raise ValueError("RGB/track frame count mismatch")
            result.update(rgb.get(frames, codec="jpeg"))
            result["state"] = "rgb-pilot"
            result["time_semantics_verified"] = rgb.metadata["time_semantics_verified"]
        elif require_rgb:
            raise FileNotFoundError("RGB requested from an annotations-only cache")
        return result

    def close(self):
        for reader in self._rgb_readers.values():
            if hasattr(reader, "close"):
                reader.close()
        self._rgb_readers.clear()
        for array in self._arrays.values():
            if hasattr(array, "_mmap"):
                array._mmap.close()
        self._arrays.clear()
