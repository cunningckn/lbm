"""Run with python -I: deny all original source/cache roots during joint reads."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
from molmo_motion_cache.common import sha256_file, write_json
from molmo_motion_cache.delivery import audit_component
from molmo_motion_cache.generic_reader import MMapMotionReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("package", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--deny", action="append", default=[])
    a = p.parse_args()
    original_available = {s: Path(s).exists() for s in a.deny}

    def hook(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(args[0])).absolute())
            if any(path == s or path.startswith(s.rstrip("/") + "/") for s in a.deny):
                raise PermissionError("original source path denied")

    sys.addaudithook(hook)
    audit = audit_component(a.package, verify_files=True)
    before = {str(p.relative_to(a.package)): sha256_file(p) for p in a.package.rglob("*") if p.is_file()}
    reader = MMapMotionReader(a.package / "numeric")
    rows = []
    for sample in reader.sample_ids:
        n = reader.clips[sample]["num_frames"]
        result = reader.get_selection(
            sample, frame_ids=[0, n // 2, n - 1], point_ids=[0], jpeg224_root=a.package / "rgb", require_rgb=True
        )
        assert result["rgb"].shape == (3, 224, 224, 3)
        assert result["intrinsics_224"].shape == (3, 3, 3)
        assert not result["time_semantics_verified"]
        assert not result["source_coordinate_convention_verified"]
        arrays = [
            "rgb",
            "points2d_source",
            "points2d_224",
            "points3d",
            "intrinsics_source_pixels",
            "intrinsics_224",
            "camera_poses_source",
            "pts",
            "frame_ids",
            "content_mask",
        ]
        rows.append(
            {
                "sample_id": sample,
                "array_sha256": {
                    k: hashlib.sha256(np.ascontiguousarray(result[k]).tobytes()).hexdigest() for k in arrays
                },
            }
        )
    reader.close()
    after = {str(p.relative_to(a.package)): sha256_file(p) for p in a.package.rglob("*") if p.is_file()}
    assert before == after
    write_json(
        a.report,
        {
            "status": "passed",
            "python_isolated": bool(sys.flags.isolated),
            "full_hash_audit": audit,
            "original_paths_available_before_deny": original_available,
            "original_paths_denied": a.deny,
            "package_unchanged": True,
            "results": rows,
        },
    )
    print("joint JPEG224 isolated migration verification passed", flush=True)


if __name__ == "__main__":
    main()
