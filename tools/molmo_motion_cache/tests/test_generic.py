from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from molmo_motion_cache.archives import build_tar_index
from molmo_motion_cache.generic import Candidate, _load_sample


def _npz_bytes(**values: object) -> bytes:
    payload = io.BytesIO()
    np.savez_compressed(payload, **values)
    return payload.getvalue()


def _add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload) if payload else None)


class MolmoSpacesEmptyTrackTest(unittest.TestCase):
    def test_zero_byte_3d_member_keeps_2d_sample_with_unavailable_3d(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracks_tar = root / "molmospaces" / "tracks" / "tracks-0000.tar"
            camera_tar = root / "molmospaces" / "camera" / "camera-0000.tar"
            tracks_tar.parent.mkdir(parents=True)
            camera_tar.parent.mkdir(parents=True)
            video_id = "zero-3d"
            points2d = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
            visibility2d = np.ones((3, 4), dtype=np.bool_)
            with tarfile.open(tracks_tar, "w:") as archive:
                _add_member(
                    archive,
                    f"tracks/{video_id}_2d.npz",
                    _npz_bytes(
                        tracks=np.array({"object": points2d}, dtype=object),
                        visibility=np.array({"object": visibility2d}, dtype=object),
                        dim=np.array([512, 512], dtype=np.int64),
                    ),
                )
                _add_member(archive, f"tracks/{video_id}_3d.npz", b"")
            with tarfile.open(camera_tar, "w:") as archive:
                _add_member(
                    archive,
                    f"camera/{video_id}.npz",
                    _npz_bytes(
                        cam_poses=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
                        intrinsics=np.eye(3, dtype=np.float32),
                    ),
                )

            track_index = build_tar_index(root, [tracks_tar], workers=1)
            camera_index = build_tar_index(root, [camera_tar], workers=1)
            candidate = Candidate(
                sample_id=f"molmospaces/object/{video_id}",
                dataset="molmospaces",
                track_kind="object",
                video_id=video_id,
                split="train",
                metadata={"num_frames": 3},
                track_members=(
                    ("track_2d", track_index[f"tracks/{video_id}_2d.npz"]),
                    ("track_3d", track_index[f"tracks/{video_id}_3d.npz"]),
                ),
                camera_members=(("camera", camera_index[f"camera/{video_id}.npz"]),),
            )

            result = _load_sample(root, candidate)
            self.assertEqual(len(result.objects), 1)
            track = result.objects[0]
            np.testing.assert_array_equal(track.points2d, points2d)
            self.assertEqual(track.points3d.shape, (3, 4, 3))
            self.assertTrue(np.isnan(track.points3d).all())
            self.assertFalse(track.visibility3d.any())


if __name__ == "__main__":
    unittest.main()
