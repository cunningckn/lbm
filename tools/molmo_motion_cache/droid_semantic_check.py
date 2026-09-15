"""Source NPZ/JSON field comparisons without calling DroidSource.load."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
from molmo_motion_cache.common import write_json
from molmo_motion_cache.droid import DroidSource
from molmo_motion_cache.reader import MMapDroidReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    reader = MMapDroidReader(args.release / "subsets/droid")
    results = []
    with DroidSource(args.source) as source:
        candidates = source.candidates()
        for split in ("train", "test"):
            group = [c for c in candidates if c.split == split]
            for candidate in (group[0], group[-1]):
                d2, d3 = source._load_npz(candidate.track_2d), source._load_npz(candidate.track_3d)
                cache = reader.get_full(candidate.sample_id)
                for name, raw in [
                    ("points2d", d2["tracks_2d"]),
                    ("points3d", d3["points_3d"]),
                    ("visibility2d", d2["visibility"]),
                    ("visibility3d", d3["valid_3d"]),
                ]:
                    np.testing.assert_array_equal(raw.astype(cache[name].dtype), cache[name])
                camera = source._load_camera_document(candidate.camera)[candidate.metadata["cam"]]
                matrix = np.asarray(camera["measured_intrinsics"], dtype="float32").reshape(3, 3)
                np.testing.assert_array_equal(matrix, cache["intrinsics_measured"])
                extrinsic = camera.get("vggt_extrinsics", camera.get("optimized_extrinsics"))
                np.testing.assert_array_equal(np.asarray(extrinsic, dtype="float32").reshape(4, 4), cache["extrinsics"])
                height, width = d2["ds_dim"]
                scale = np.diag([width / (2 * matrix[0, 2]), height / (2 * matrix[1, 2]), 1])
                np.testing.assert_allclose((scale @ matrix).astype("float32"), cache["intrinsics_ds"], rtol=1e-6)
                results.append(
                    {
                        "split": split,
                        "sample_id": candidate.sample_id,
                        "status": "passed",
                        "dtype2d": str(d2["tracks_2d"].dtype),
                        "dtype3d": str(d3["points_3d"].dtype),
                    }
                )
    write_json(
        args.report,
        {
            "status": "passed",
            "results": results,
            "scope": "first/last train/test entries, full arrays and source camera fields",
        },
    )
    print("DROID independent parity passed", flush=True)


if __name__ == "__main__":
    main()
