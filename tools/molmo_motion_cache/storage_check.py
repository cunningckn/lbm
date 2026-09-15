"""Fixed-source-manifest accounting, excluding failed and download partials."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from molmo_motion_cache.common import checksum_entries, read_json, write_json
from molmo_motion_cache.release import SOURCE_DATASETS


def measure(paths):
    stats = [p.stat() for p in paths]
    unique = {(s.st_dev, s.st_ino): s for s in stats}
    return {
        "files": len(stats),
        "logical_bytes": sum(s.st_size for s in stats),
        "allocated_bytes": sum(s.st_blocks * 512 for s in stats),
        "inode_deduplicated_allocated_bytes": sum(s.st_blocks * 512 for s in unique.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("release", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    trees = list((args.source / ".cache/huggingface/trees").glob("*.json"))
    assert len(trees) == 1
    source_paths = [args.source / name for name in read_json(trees[0])["files"]]
    delivery_paths = []
    for component in [*(args.release / "subsets" / s for s in SOURCE_DATASETS), args.release / "assets"]:
        delivery_paths.extend(component / name for name in [*checksum_entries(component), "READY.json", "SHA256SUMS"])
    raw, delivered, union = measure(source_paths), measure(delivery_paths), measure(source_paths + delivery_paths)
    result = {
        "source_revision": trees[0].stem,
        "source_fixed_snapshot": raw,
        "published_components": delivered,
        "raw_plus_delivery": union,
        "incremental_allocated_bytes": union["inode_deduplicated_allocated_bytes"]
        - raw["inode_deduplicated_allocated_bytes"],
        "scope": "only pinned source manifest and component manifests; excludes incomplete downloads, jobs and pilots",
    }
    write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
