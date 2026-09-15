"""Independent field comparisons; shares discovery, never builder normalization."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
from molmo_motion_cache.archives import read_npz
from molmo_motion_cache.common import write_json
from molmo_motion_cache.generic import GENERIC_SUBSETS, candidate_seeds, resolve_candidates
from molmo_motion_cache.generic_reader import MMapMotionReader


def mapping(value):
    return {str(k): np.asarray(v) for k, v in value.item().items()}


def check(source, release):
    results = []
    for subset in GENERIC_SUBSETS:
        reader = MMapMotionReader(release / "subsets" / subset)
        candidates = resolve_candidates(
            source, subset, candidate_seeds(source, subset, limit_per_track_kind=2), workers=2
        )
        for candidate in candidates:
            docs = {role: read_npz(source, ref) for role, ref in candidate.track_members}
            pairs = []
            if subset == "xperience":
                for role, doc in docs.items():
                    pairs.append(
                        (
                            role,
                            doc["tracks_2d"] if role == "object" else doc["pixel_coords"],
                            doc["points_3d"],
                            doc.get("visibility_2d"),
                            doc.get("visibility"),
                            True,
                        )
                    )
            else:
                d2, d3 = docs["track_2d"], docs["track_3d"]
                if subset == "egodex" and candidate.track_kind == "object":
                    pairs = [
                        ("object", d2["tracks"], d3["points_3d"], d2.get("visibility"), d3.get("visibility"), False)
                    ]
                else:
                    p3 = mapping(d3["points_3d"])
                    v3 = mapping(d3["visibility"]) if "visibility" in d3 else {}
                    offset = 0
                    for role, p in p3.items():
                        if subset == "hdepic":
                            p2 = d2["tracks"][:, offset : offset + len(p)]
                            v2 = d2["visibility"][:, offset : offset + len(p)]
                            offset += len(p)
                        else:
                            p2, v2 = mapping(d2["tracks"])[role], mapping(d2["visibility"]).get(role)
                        pairs.append((role, p2, p, v2, v3.get(role), False))
            dtype_records = []
            for role, p2, p3, v2, v3, transpose2 in pairs:
                cached = reader.get_object_full(candidate.sample_id, role)
                for field, raw, transpose, mask in [("points2d", p2, transpose2, v2), ("points3d", p3, True, v3)]:
                    dtype_records.append({"field": field, "dtype": str(raw.dtype)})
                    expected = raw.swapaxes(0, 1) if transpose else raw
                    np.testing.assert_array_equal(expected.astype("float32"), cached[field])
                    if mask is None:
                        visible = np.isfinite(expected).all(axis=-1)
                    else:
                        visible = np.asarray(mask)
                        if visible.ndim == 3:
                            visible = visible[..., 0]
                        if transpose:
                            visible = visible.T
                    np.testing.assert_array_equal(visible.astype(bool), cached[field.replace("points", "visibility")])
                if subset == "xperience":
                    doc = docs[role]
                    if "trust_weights" in doc:
                        np.testing.assert_array_equal(doc["trust_weights"].T.astype("float32"), cached["trust_weights"])
                    if "keep_mask" in doc:
                        np.testing.assert_array_equal(doc["keep_mask"].astype(bool), cached["keep_mask"])
            for role, ref in candidate.camera_members:
                doc = read_npz(source, ref)
                if role == "camera":
                    np.testing.assert_array_equal(doc["cam_poses"].astype("float32"), cached["camera_poses"])
                    np.testing.assert_array_equal(
                        doc["intrinsics"][None].astype("float32"), cached["camera_intrinsics_static"]
                    )
                else:
                    data_field, index_field = (
                        ("camera_poses", "camera_pose_indices")
                        if role == "camera_pose"
                        else ("camera_intrinsics_dynamic", "camera_intrinsic_indices")
                    )
                    np.testing.assert_array_equal(doc["data"].astype("float32"), cached[data_field])
                    np.testing.assert_array_equal(doc.get("inds", np.arange(len(doc["data"]))), cached[index_field])
            results.append(
                {
                    "subset": subset,
                    "split": candidate.split,
                    "kind": candidate.track_kind,
                    "sample_id": candidate.sample_id,
                    "objects": len(pairs),
                    "raw_dtypes": dtype_records,
                    "status": "passed",
                }
            )
        print(subset, "passed", flush=True)
    return {
        "status": "passed",
        "scope": ("two source entries per split/kind; full arrays including first/last frame; "
                  "independent field normalization"),
        "shared_with_builder": "candidate discovery and tar byte reader only",
        "not_covered": ["DROID independent normalization", "projection geometry", "entire corpus semantic equivalence"],
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    write_json(args.report, check(args.source, args.release))
