from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tc_cloud" / "submit_molmo_motion_full.py"
SPEC = importlib.util.spec_from_file_location("submit_molmo_motion_full", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SUBMIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBMIT)


class CloudSubmitRequestTest(unittest.TestCase):
    def test_request_is_cpu_only_with_exactly_100_cores(self) -> None:
        args = argparse.Namespace(
            workers=100,
            shard_size=256,
            checks=32,
            memory_gb=625,
            resource_group_id="test-group",
            image_url="registry/image:test",
            source_root=Path("/data/raw"),
            output_root=Path("/data/output"),
        )
        params = SUBMIT.build_params(
            args,
            {"USER_NAME": "tester"},
            "test-job",
            Path("/code"),
            Path("/data/output/_jobs/test-job/build.log"),
        )
        resource = params["ResourceConfigInfos"][0]
        self.assertEqual(resource["Cpu"], 100_000)
        self.assertEqual(resource["Gpu"], 0)
        self.assertEqual(resource["GpuType"], "")
        self.assertEqual(resource["InstanceNum"], 1)
        command = params["StartCmdInfo"]["StartCmd"]
        self.assertIn("WORKERS=100", command)
        self.assertIn("SHARD_SIZE=256", command)
        self.assertIn("MODE=full", command)
        self.assertIn("VERIFY_SOURCE_HASHES=1", command)


if __name__ == "__main__":
    unittest.main()
