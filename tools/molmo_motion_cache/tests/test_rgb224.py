import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from molmo_motion_cache.common import read_json, write_json
from molmo_motion_cache.rgb224 import FRAME_DTYPE, JPEG224Reader, build_group, publish
from PIL import Image


class ShardReaderTest(unittest.TestCase):
    def fixture(self, root):
        group = root / "group-000"
        group.mkdir()
        rows = []
        for i in range(3):
            buffer = io.BytesIO()
            Image.new("RGB", (224, 224), (i * 70, 20, 40)).save(buffer, "JPEG")
            payload = buffer.getvalue()
            (group / f"frames-{i:05d}.bin").write_bytes(payload)
            rows.append((i, 0, len(payload), i, 1280 * i))
        np.save(group / "frames.npy", np.array(rows, dtype=FRAME_DTYPE))
        write_json(
            group / "dataset.json",
            {
                "contract": {"test": True},
                "videos": {"v": {"start": 0, "count": 3, "time_base": [1, 19392], "geometry": {}}},
            },
        )
        publish(group)
        write_json(
            root / "dataset.json", {"format_version": 1, "state": "jpeg224-frame-index-pilot", "groups": ["group-000"]}
        )
        publish(root)

    def test_cross_shard_duplicate_frames_lru_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with JPEG224Reader(root, max_open_files=1) as reader:
                result = reader.get("v", [2, 0, 2, 1])
                self.assertEqual(result["rgb"].shape, (4, 224, 224, 3))
                np.testing.assert_array_equal(result["rgb"][0], result["rgb"][2])
                np.testing.assert_array_equal(result["pts"], [2560, 0, 2560, 1280])
                self.assertEqual(len(reader.handles), 1)
                for ids in [[], [-1], [3], [0.5]]:
                    with self.assertRaises(IndexError):
                        reader.get("v", ids)

    def test_resume_refuses_wrong_contract_or_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                build_group((str(root), str(root / "group-000"), [], {}))
            with (root / "group-000/frames-00000.bin").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build_group((str(root), str(root / "group-000"), [], {}))

    def test_unpublished_and_unknown_version_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                JPEG224Reader(root)
            self.fixture(root)
            doc = read_json(root / "dataset.json")
            doc["format_version"] = 99
            write_json(root / "dataset.json", doc)
            publish(root)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                JPEG224Reader(root)


if __name__ == "__main__":
    unittest.main()
