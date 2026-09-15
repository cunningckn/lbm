from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from molmo_motion_cache.common import write_checksum_manifest, write_json, write_parquet
from molmo_motion_cache.droid import DroidSource
from molmo_motion_cache.reader import MMapDroidReader


class ReaderLayoutTest(unittest.TestCase):
    def test_scaled_intrinsics_matches_target_geometry(self) -> None:
        measured = np.array(
            [[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        actual = DroidSource.scaled_intrinsics(measured, np.array([480, 854], dtype=np.int64))
        expected = np.array(
            [[854.0, 0.0, 427.0], [0.0, 720.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_reader_reconstructs_and_windows_flattened_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shards" / "droid" / "000000"
            shard.mkdir(parents=True)
            np.save(shard / "points2d.npy", np.arange(12, dtype=np.float32).reshape(6, 2))
            np.save(shard / "points3d.npy", np.arange(18, dtype=np.float32).reshape(6, 3))
            np.save(shard / "visibility2d.npy", np.ones(6, dtype=np.bool_))
            np.save(shard / "visibility3d.npy", np.zeros(6, dtype=np.bool_))
            np.save(
                shard / "camera_intrinsics_measured.npy",
                np.eye(3, dtype=np.float32)[None],
            )
            np.save(shard / "camera_intrinsics_ds.npy", np.eye(3, dtype=np.float32)[None])
            np.save(shard / "camera_extrinsics.npy", np.eye(4, dtype=np.float32)[None])
            write_json(
                root / "dataset.json",
                {
                    "format": "molmo-motion-cache",
                    "format_version": 1,
                    "subset": "droid",
                    "runtime_contract": {"all_runtime_paths_relative": True},
                },
            )
            track = {
                "sample_id": "droid/test",
                "shard": "shards/droid/000000",
                "row_offset": 0,
                "row_count": 6,
                "num_frames": 2,
                "num_points": 3,
            }
            write_parquet([track], root / "tracks_index.parquet")
            write_parquet(
                [{"sample_id": "droid/test", "shard": "shards/droid/000000", "camera_row": 0}],
                root / "cameras_index.parquet",
            )
            write_parquet([{"sample_id": "droid/test"}], root / "clips.parquet")
            with self.assertRaises((ValueError, FileNotFoundError)):
                MMapDroidReader(root)
            write_json(
                root / "READY.json",
                {"format_version": 1, "status": "ready", "sha256sums_sha256": write_checksum_manifest(root)},
            )
            reader = MMapDroidReader(root)
            full = reader.get_full("droid/test")
            self.assertEqual(full["points3d"].shape, (2, 3, 3))
            window = reader.get_window("droid/test", start=0, frames=2, points=2)
            self.assertEqual(window["points2d"].shape, (2, 2, 2))
            np.testing.assert_array_equal(
                window["points2d"],
                np.array([[[0.0, 1.0], [4.0, 5.0]], [[6.0, 7.0], [10.0, 11.0]]]),
            )
            write_json(root / "dataset.json", {"format_version": 99})
            with self.assertRaises(ValueError):
                MMapDroidReader(root)


if __name__ == "__main__":
    unittest.main()
