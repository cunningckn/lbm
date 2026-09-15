import tempfile
import unittest
from pathlib import Path

import numpy as np
from molmo_motion_cache.common import write_checksum_manifest, write_json


class RGBSchemaTest(unittest.TestCase):
    def test_unknown_rgb_metadata_version_is_rejected(self):
        try:
            from molmo_motion_cache.rgb_pilot import RGBReader
        except ImportError:
            self.skipTest("optional RGB dependencies are not installed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / 'frames.npy', np.zeros(1, dtype='uint64'))
            write_json(root / 'dataset.json', {'format_version': 99, 'state': 'rgb-pilot'})
            write_json(root / 'PILOT_READY.json', {'format_version': 1, 'status': 'pilot-ready',
                       'sha256sums_sha256': write_checksum_manifest(root)})
            with self.assertRaisesRegex(ValueError, 'unsupported RGB'):
                RGBReader(root)
