from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molmo_motion_cache.common import (
    verify_checksum_manifest,
    write_checksum_manifest,
    write_completion_marker,
    write_json,
)
from molmo_motion_cache.generic import build_generic_cache
from molmo_motion_cache.generic_reader import MMapMotionReader
from molmo_motion_cache.identity import (
    component_identity,
    load_source_snapshot,
    with_paired_component_fingerprints,
)
from molmo_motion_cache.release import build_portable_assets


def _npz_bytes(**values: object) -> bytes:
    payload = io.BytesIO()
    np.savez_compressed(payload, **values)
    return payload.getvalue()


def _add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload) if payload else None)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_tree_manifest(source: Path) -> None:
    files = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
        files[relative] = {"size": path.stat().st_size}
    _write_json(source / ".cache" / "huggingface" / "trees" / "synthetic.json", {"files": files})


def _world_to_pixels(points: np.ndarray, c2w: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((*points.shape[:2], 1))], axis=-1)
    world_to_camera = np.linalg.inv(c2w)
    camera = np.einsum("tij,tkj->tki", world_to_camera, homogeneous)[..., :3]
    projected = camera @ intrinsics.T
    return projected[..., :2] / projected[..., 2:3]


def _write_molmospaces_source(source: Path) -> dict[str, np.ndarray | str]:
    video_id = "moving-camera"
    frames, points = 3, 3
    intrinsics = np.array([[120.0, 0.0, 64.0], [0.0, 120.0, 48.0], [0.0, 0.0, 1.0]])
    points3d = np.array(
        [
            [[1.0, 0.0, 5.0], [1.5, 0.5, 6.0], [0.5, -0.5, 4.0]],
            [[1.2, 0.1, 5.0], [1.4, 0.6, 6.0], [0.6, -0.4, 4.0]],
            [[1.4, 0.2, 5.0], [1.3, 0.7, 6.0], [0.7, -0.3, 4.0]],
        ],
        dtype=np.float32,
    )
    camera_to_world = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    camera_to_world[:, 0, 3] = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    points2d = _world_to_pixels(points3d, camera_to_world, intrinsics).astype(np.float32)
    visibility = np.ones((frames, points), dtype=np.bool_)
    metadata = {
        "file": video_id,
        "num_frames": frames,
        "fps": 15.0,
        "caption": "synthetic moving camera",
        "clips_by_object": {"object": [[0, frames - 1]]},
    }
    _write_json(
        source / "molmospaces" / "annotations" / "molmospaces_split.json",
        {"train": [metadata], "test": []},
    )
    _write_json(source / "molmospaces" / "annotations" / "molmospaces_clips.json", [metadata])
    tracks = source / "molmospaces" / "tracks" / "tracks-0000.tar"
    cameras = source / "molmospaces" / "camera" / "camera-0000.tar"
    tracks.parent.mkdir(parents=True)
    cameras.parent.mkdir(parents=True)
    with tarfile.open(tracks, "w:") as archive:
        _add_member(
            archive,
            f"tracks/{video_id}_2d.npz",
            _npz_bytes(
                tracks=np.array({"object": points2d}, dtype=object),
                visibility=np.array({"object": visibility}, dtype=object),
                dim=np.array([96, 128], dtype=np.int64),
            ),
        )
        _add_member(
            archive,
            f"tracks/{video_id}_3d.npz",
            _npz_bytes(
                points_3d=np.array({"object": points3d.transpose(1, 0, 2)}, dtype=object),
                visibility=np.array({"object": visibility.T}, dtype=object),
            ),
        )
    with tarfile.open(cameras, "w:") as archive:
        _add_member(
            archive,
            f"camera/{video_id}.npz",
            _npz_bytes(cam_poses=camera_to_world, intrinsics=intrinsics.astype(np.float32)),
        )
    videos = source / "molmospaces" / "videos" / "videos-0000.tar"
    robots = source / "molmospaces" / "robot_trajectories" / "robot_trajectories-0000.tar"
    videos.parent.mkdir(parents=True)
    robots.parent.mkdir(parents=True)
    with tarfile.open(videos, "w:") as archive:
        _add_member(archive, f"videos/{video_id}.mp4", b"synthetic-mp4")
    with tarfile.open(robots, "w:") as archive:
        _add_member(archive, "robot_trajectories/scenario__house.h5", b"synthetic-h5")
    return {
        "video_id": video_id,
        "points2d": points2d,
        "points3d": points3d,
        "camera_to_world": camera_to_world,
        "intrinsics": intrinsics,
    }


class GenericBuildContractTest(unittest.TestCase):
    def test_molmospaces_build_is_portable_and_keeps_c2w_pose_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            values = _write_molmospaces_source(source)
            _write_tree_manifest(source)
            snapshot = load_source_snapshot(source)
            assets_identity = component_identity(snapshot, "assets")
            numeric_identity = with_paired_component_fingerprints(
                component_identity(snapshot, "molmospaces"),
                {"assets": assets_identity["component_fingerprint"]},
            )
            output = root / "numeric"
            build_generic_cache(
                source,
                output,
                "molmospaces",
                limit_per_track_kind=None,
                shard_size=1,
                workers=1,
                checks=1,
                checksums=True,
                source_identity=numeric_identity,
            )
            assets = root / "assets"
            build_portable_assets(source, assets, workers=1, source_identity=assets_identity)

            reader = MMapMotionReader(output)
            sample_id = f"molmospaces/object/{values['video_id']}"
            full = reader.get_object_full(sample_id, "object")
            np.testing.assert_array_equal(full["points2d"], values["points2d"])
            np.testing.assert_array_equal(full["points3d"], values["points3d"])
            np.testing.assert_array_equal(full["camera_poses"], values["camera_to_world"])
            projected = _world_to_pixels(
                full["points3d"], full["camera_poses"], full["camera_intrinsics_static"][0]
            )
            np.testing.assert_allclose(projected, full["points2d"], atol=1e-4)
            direct = full["points3d"] @ full["camera_intrinsics_static"][0].T
            direct = direct[..., :2] / direct[..., 2:3]
            self.assertGreater(float(np.max(np.abs(direct - full["points2d"]))), 1.0)
            self.assertEqual(reader.validate_paired_assets(assets)["secondary_component"], "assets")
            self.assertGreater(verify_checksum_manifest(output)["checked_files"], 0)

            wrong_assets = root / "wrong-assets"
            wrong_assets.mkdir()
            wrong_identity = dict(assets_identity)
            wrong_identity["snapshot_revision"] = "other-snapshot"
            wrong_identity["snapshot_fingerprint"] = "a" * 64
            write_json(
                wrong_assets / "dataset.json",
                {
                    "format": "molmo-motion-portable-assets",
                    "source_identity": wrong_identity,
                },
            )
            (wrong_assets / "payload.bin").write_bytes(b"same-video-id-and-size")
            digest = write_checksum_manifest(wrong_assets)
            write_completion_marker(
                wrong_assets,
                {"status": "ready", "sha256sums_sha256": digest},
                pilot=False,
            )
            with self.assertRaisesRegex(ValueError, "different source snapshots"):
                reader.validate_paired_assets(wrong_assets)

            copied = root / "copied-numeric"
            shutil.copytree(output, copied)
            shutil.rmtree(source)
            copied_reader = MMapMotionReader(copied)
            window = copied_reader.get_object_window(sample_id, "object", frames=2, points=2)
            self.assertEqual(window["points2d"].shape, (2, 2, 2))
            np.testing.assert_array_equal(window["source_point_indices"], np.array([0, 2]))

    def test_xperience_selection_keeps_weights_and_original_point_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            video_id = "selected-object"
            frames = 2
            keep_mask = np.array([True, False, True, True, False])
            source_indices = np.flatnonzero(keep_mask)
            points2d = np.arange(12, dtype=np.float32).reshape(3, frames, 2)
            points3d = np.arange(18, dtype=np.float32).reshape(3, frames, 3)
            weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
            metadata = {
                "file": video_id,
                "num_frames": frames,
                "fps": 10.0,
                "caption": "synthetic selection",
                "clips_by_object": {"object": [[0, 1]]},
            }
            _write_json(
                source / "xperience" / "annotations" / "xperience_split.json",
                {"train": [metadata], "test": []},
            )
            _write_json(
                source / "xperience" / "annotations" / "xperience_hand_split.json",
                {"train": [], "test": []},
            )
            tar_path = source / "xperience" / "tracks" / "tracks-0000.tar"
            tar_path.parent.mkdir(parents=True)
            with tarfile.open(tar_path, "w:") as archive:
                _add_member(
                    archive,
                    f"tracks/{video_id}_object.npz",
                    _npz_bytes(
                        tracks_2d=points2d,
                        points_3d=points3d,
                        visibility_2d=np.ones((3, frames), dtype=np.bool_),
                        visibility=np.ones((3, frames), dtype=np.bool_),
                        trust_weights=weights,
                        keep_mask=keep_mask,
                    ),
                )
            output = root / "xperience-pilot"
            build_generic_cache(
                source,
                output,
                "xperience",
                limit_per_track_kind=1,
                shard_size=1,
                workers=1,
                checks=1,
                checksums=True,
            )
            reader = MMapMotionReader(output)
            sample_id = f"xperience/object/{video_id}"
            full = reader.get_object_full(sample_id, "object")
            np.testing.assert_array_equal(full["points2d"], points2d.transpose(1, 0, 2))
            np.testing.assert_array_equal(full["trust_weights"], weights.T)
            np.testing.assert_array_equal(full["source_point_indices"], source_indices)
            row = reader.tracks[sample_id][0]
            self.assertEqual(row["source_point_indices"], source_indices.tolist())
            window = reader.get_object_window(sample_id, "object", frames=2, points=2)
            np.testing.assert_array_equal(window["clip_frame_indices"], np.array([0, 1]))
            np.testing.assert_array_equal(window["source_point_indices"], np.array([0, 3]))
            np.testing.assert_array_equal(window["trust_weights"], weights.T[:, [0, 2]])


if __name__ == "__main__":
    unittest.main()
