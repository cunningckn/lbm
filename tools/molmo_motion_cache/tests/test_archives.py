from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from molmo_motion_cache.archives import build_tar_index, read_npz
from molmo_motion_cache.generic import _track_object


class ArchiveReaderTest(unittest.TestCase):
    def test_npz_member_is_read_directly_by_tar_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "subset" / "tracks" / "tracks-0000.tar"
            shard.parent.mkdir(parents=True)
            payload = io.BytesIO()
            np.savez_compressed(
                payload,
                points=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
                labels=np.array({"left": [1, 2]}, dtype=object),
            )
            data = payload.getvalue()
            with tarfile.open(shard, mode="w:") as archive:
                member = tarfile.TarInfo("tracks/example.npz")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

            index = build_tar_index(
                root,
                [shard],
                workers=1,
                wanted_names={"tracks/example.npz"},
            )
            document = read_npz(root, index["tracks/example.npz"])
            np.testing.assert_array_equal(document["points"], np.arange(18, dtype=np.float32).reshape(2, 3, 3))
            self.assertEqual(document["labels"].item(), {"left": [1, 2]})

    def test_track_axes_are_normalized_to_time_point_dimension(self) -> None:
        points2d = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
        points3d = np.arange(36, dtype=np.float32).reshape(3, 4, 3)
        visibility = np.ones((3, 4), dtype=np.bool_)
        result = _track_object(
            "hand",
            points2d,
            points3d,
            visibility,
            visibility,
            points2d_order="KT",
            points3d_order="KT",
        )
        self.assertEqual(result.points2d.shape, (4, 3, 2))
        self.assertEqual(result.points3d.shape, (4, 3, 3))
        self.assertEqual(result.visibility2d.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
