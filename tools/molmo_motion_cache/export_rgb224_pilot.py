"""Independent small joint package. Never copy the full numerical shards."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from molmo_motion_cache.common import read_json, write_json, write_parquet
from molmo_motion_cache.delivery import export_component
from molmo_motion_cache.generic import _write_shard, candidate_seeds, resolve_candidates
from molmo_motion_cache.rgb224 import JPEG224Reader, publish


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("release", type=Path)
    p.add_argument("rgb", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    if a.destination.exists():
        raise FileExistsError(a.destination)
    a.destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".joint224-", dir=a.destination.parent))
    with JPEG224Reader(a.rgb) as reader:
        videos = set(reader.videos)
    if len(videos) > 4:
        raise ValueError("small migration fixture must contain at most four videos")
    seeds = [s for s in candidate_seeds(a.source, "molmospaces", limit_per_track_kind=None) if s.video_id in videos]
    if {s.video_id for s in seeds} != videos:
        raise ValueError("missing numeric sample")
    candidates = resolve_candidates(a.source, "molmospaces", seeds, workers=1)
    numeric = staging / "numeric"
    numeric.mkdir()
    result = _write_shard(str(a.source), str(numeric), "molmospaces", 0, candidates)
    for rows, name in [
        (result.clips, "clips.parquet"),
        (result.tracks, "tracks_index.parquet"),
        (result.cameras, "cameras_index.parquet"),
        (result.motion_ranges, "motion_ranges.parquet"),
    ]:
        write_parquet(rows, numeric / name)
    metadata = read_json(a.release / "subsets/molmospaces/dataset.json")
    metadata.update(records=len(seeds), build_scope="jpeg224-migration-pilot", trajectory_rows=result.trajectory_rows)
    write_json(numeric / "dataset.json", metadata)
    publish(numeric)
    export_component(a.rgb, staging / "rgb")
    write_json(
        staging / "dataset.json",
        {
            "format_version": 1,
            "state": "joint-jpeg224-pilot",
            "numeric": "numeric",
            "rgb": "rgb",
            "source_coordinate_convention_verified": False,
            "time_semantics_verified": False,
            "full_multimodal": False,
        },
    )
    publish(staging)
    os.rename(staging, a.destination)
    print(a.destination)


if __name__ == "__main__":
    main()
