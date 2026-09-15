"""Independent process check: original data opens are actively denied."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from molmo_motion_cache.delivery import audit_component
from molmo_motion_cache.reader import MMapDroidReader

parser = argparse.ArgumentParser()
parser.add_argument("copy", type=Path)
args = parser.parse_args()
forbidden = (
    "/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m/",
    "/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1/",
)


def deny_original(event, values):
    if event == "open" and isinstance(values[0], (str, bytes)):
        path = str(Path(values[0]).absolute())
        if path.startswith(forbidden):
            raise PermissionError("original dataset disabled in relocation test")


sys.addaudithook(deny_original)
sys.dont_write_bytecode = True
before = audit_component(args.copy, verify_files=True)
reader = MMapDroidReader(args.copy)
result = reader.get_window(reader.sample_ids[0], frames=8, points=32)
after = audit_component(args.copy, verify_files=True)
assert before == after
print(
    json.dumps(
        {
            "status": "passed",
            "original_path_access": "denied-by-audit-hook",
            "isolated_python": sys.flags.isolated,
            "samples": len(reader.sample_ids),
            "window_shape": result["points2d"].shape,
            "copy_unchanged": True,
        }
    )
)
