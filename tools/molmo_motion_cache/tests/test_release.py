from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from molmo_motion_cache.common import read_completion_marker, write_checksum_manifest, write_json
from molmo_motion_cache.release import SOURCE_DATASETS, build_stereo4d_metadata, verify_release


class ReleaseVerificationTest(unittest.TestCase):
    def test_all_component_manifests_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components = [*(root / "subsets" / name for name in SOURCE_DATASETS), root / "assets"]
            for component in components:
                component.mkdir(parents=True)
                (component / "payload.bin").write_bytes(b"verified payload")
                digest = write_checksum_manifest(component)
                write_json(
                    component / "READY.json",
                    {"status": "ready", "sha256sums_sha256": digest},
                )
            write_json(root / "dataset.json", {"format": "molmo-motion-cache-release"})
            write_json(root / "READY.json", {"status": "ready"})

            result = verify_release(root, verify_files=True)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(set(result["components"]), {*SOURCE_DATASETS, "assets"})
            self.assertTrue(
                all(
                    value["status"] == "files-verified"
                    for value in result["components"].values()
                )
            )
            self.assertEqual(result["source_identity"]["status"], "legacy-unverified")
            with self.assertRaisesRegex(ValueError, "predates source identity"):
                verify_release(root, verify_files=False, require_source_identity=True)

    def test_stereo4d_metadata_uses_the_shared_full_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            video_id = "metadata-only"
            write_json(
                source / "stereo4d" / "annotations" / "stereo4d_split.json",
                {
                    "train": [
                        {
                            "file": video_id,
                            "fps": 12.0,
                            "num_frames": 2,
                            "caption": "synthetic Stereo4D metadata",
                            "clips_by_object": {"object": [[0, 1]]},
                        }
                    ],
                    "test": [],
                },
            )
            write_json(
                source / "stereo4d" / "track_index" / "stereo4d_track_index.json",
                {"clips": {video_id: {"objects": {"object": [0, 1]}, "T0": []}}},
            )
            output = Path(temporary) / "stereo4d"
            build_stereo4d_metadata(source, output, limit=None)
            marker, metadata = read_completion_marker(output, pilot=False)
            self.assertEqual(marker.name, "READY.json")
            self.assertEqual(metadata["status"], "metadata-only-ready")


if __name__ == "__main__":
    unittest.main()
