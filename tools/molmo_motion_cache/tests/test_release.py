from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from molmo_motion_cache.common import write_checksum_manifest, write_json
from molmo_motion_cache.release import SOURCE_DATASETS, verify_release


class ReleaseVerificationTest(unittest.TestCase):
    def test_all_component_manifests_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components = [*(root / "subsets" / name for name in SOURCE_DATASETS), root / "assets"]
            for component in components:
                component.mkdir(parents=True)
                (component / "payload.bin").write_bytes(b"verified payload")
                digest = write_checksum_manifest(component)
                write_json(component / "READY.json", {"sha256sums_sha256": digest})
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


if __name__ == "__main__":
    unittest.main()
