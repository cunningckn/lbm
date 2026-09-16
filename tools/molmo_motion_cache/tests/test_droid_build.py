import io
import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from molmo_motion_cache.common import verify_checksum_manifest
from molmo_motion_cache.droid import build_droid_cache
from molmo_motion_cache.reader import MMapDroidReader


def test_droid_build_validates_before_publishing():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        for folder in ("annotations", "tracks", "camera"):
            (source / "droid" / folder).mkdir(parents=True)
        metadata = {"file": "scene__cam", "cam": "cam", "num_frames": 2, "ds_dim": [8, 10]}
        metadata["clips_by_object"] = {"object": [[0, 1]]}
        for name in ("droid_clips.json", "droid_videos_index.json"):
            (source / "droid/annotations" / name).write_text("{}")
        (source / "droid/annotations/droid_split.json").write_text(
            json.dumps({"train": [metadata], "test": []})
        )
        points = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        with tarfile.open(source / "droid/tracks/tracks-0000.tar", "w") as archive:
            for kind, arrays in (
                ("2d", dict(tracks_2d=points, visibility=np.ones((2, 3)), ds_dim=[8, 10])),
                ("3d", dict(points_3d=np.ones((2, 3, 3), dtype=np.float32), valid_3d=np.ones((2, 3)))),
            ):
                payload = io.BytesIO()
                np.savez(payload, **arrays)
                member = tarfile.TarInfo(f"tracks/scene__cam_{kind}.npz")
                member.size = len(payload.getvalue())
                archive.addfile(member, io.BytesIO(payload.getvalue()))
        calibration = {"cam": {
            "measured_intrinsics": [[10, 0, 5], [0, 10, 4], [0, 0, 1]],
            "vggt_extrinsics": np.eye(4).tolist(),
        }}
        with tarfile.open(source / "droid/camera/camera-0000.tar", "w") as archive:
            payload = json.dumps(calibration).encode()
            member = tarfile.TarInfo("camera/scene_cameras.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        original = MMapDroidReader._open_staging

        def check_staging(path):
            assert not (path / "READY.json").exists()
            assert not (path / "PILOT_READY.json").exists()
            try:
                MMapDroidReader(path)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("public reader accepted unfinished output")
            return original(path)

        for limit in (None, 1):
            output = root / str(limit)
            with patch.object(MMapDroidReader, "_open_staging", side_effect=check_staging):
                build_droid_cache(source, output, limit=limit, shard_size=1, checks=1)
            reader = MMapDroidReader(output)
            np.testing.assert_array_equal(reader.get_full(reader.sample_ids[0])["points2d"], points)
            assert verify_checksum_manifest(output)["checked_files"] > 0
