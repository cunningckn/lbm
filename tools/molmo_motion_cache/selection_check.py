import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
from molmo_motion_cache.common import write_json
from molmo_motion_cache.generic_reader import MMapMotionReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("pilot", type=Path)
    args = parser.parse_args()
    reader = MMapMotionReader(args.release / "subsets/molmospaces")
    results = []
    for directory in sorted(args.pilot.glob("*/PILOT_READY.json")):
        sample = "molmospaces/object/" + directory.parent.name
        frames = [0, reader.clips[sample]["num_frames"] - 1]
        result = reader.get_selection(sample, frame_ids=frames, point_ids=[0], rgb_root=args.pilot, require_rgb=True)
        np.testing.assert_array_equal(result["frame_ids"], frames)
        assert result["rgb"].shape[0] == result["points2d"].shape[0] == 2
        try:
            reader.get_selection(sample, frame_ids=frames, point_ids=[0], require_rgb=True)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("RGB missing must fail")
        for ids in ([-1], [reader.clips[sample]["num_frames"]], [0.5]):
            try:
                reader.get_selection(sample, frame_ids=ids, point_ids=[0])
            except IndexError:
                pass
            else:
                raise AssertionError("invalid frame selection accepted")
        results.append(
            {"sample": sample, "state": result["state"], "time_semantics_verified": result["time_semantics_verified"]}
        )
    write_json(args.pilot / "SELECTION_CHECK.json", {"status": "passed", "results": results})
    print("explicit RGB/numeric selection and missing-RGB checks passed")


if __name__ == "__main__":
    main()
