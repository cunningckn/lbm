"""Matched persistent independent-reader benchmark (not a training DataLoader)."""

import argparse
import hashlib
import json
import multiprocessing
import resource
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import av
import numpy as np
from molmo_motion_cache.common import write_json
from molmo_motion_cache.generic_reader import MMapMotionReader
from molmo_motion_cache.geometry224 import content_mask, resize224, select_camera, transform_points
from molmo_motion_cache.rgb224 import JPEG224Reader
from molmo_motion_cache.rgb_pilot import MemberIO


def partition(spec):
    release, cache, mode, operations = spec
    start = time.perf_counter()
    numeric = MMapMotionReader(Path(release) / "subsets/molmospaces")
    rgb = JPEG224Reader(cache)
    containers = OrderedDict()
    init = time.perf_counter() - start
    latencies, checksums = [], []
    steady_start = time.perf_counter()
    for video_id, ids, points in operations:
        tick = time.perf_counter()
        sample = "molmospaces/object/" + video_id
        if mode == "jpeg85":
            # Reuse this worker's reader, avoiding a second initialization.
            numeric._rgb_readers[("jpeg224", str(Path(cache).resolve()))] = rgb
            result = numeric.get_selection(sample, frame_ids=ids, point_ids=points, jpeg224_root=cache)
            value = result["rgb"]
        else:
            _, video = rgb.videos[video_id]
            group, _ = rgb.videos[video_id]
            frames = rgb.indices[group][video["start"] + np.array(ids)]
            row = video["asset"]
            pair = containers.pop(video_id, None)
            if pair is None:
                member = MemberIO(Path(release) / "assets" / row["archive"], row["offset"], row["size"])
                container = av.open(member)
                container.streams.video[0].thread_count = 1
                pair = member, container
            containers[video_id] = pair
            while len(containers) > 16:
                m, c = containers.popitem(last=False)[1]
                c.close()
                m.close()
            member, container = pair
            targets = {int(p): i for i, p in enumerate(frames["pts"])}
            container.seek(min(targets), stream=container.streams.video[0], backward=True, any_frame=False)
            found = {}
            for frame in container.decode(video=0):
                if frame.pts in targets:
                    found[frame.pts] = resize224(
                        frame.to_ndarray(format="rgb24"), source_size=video["geometry"]["source_size"]
                    )[0]
                if frame.pts >= max(targets):
                    break
            value = np.stack([found[int(p)] for p in frames["pts"]])
            result = numeric.get_selection(sample, frame_ids=ids, point_ids=points)
            poses, k = select_camera(numeric.get_object_full(sample), ids)
            result.update(
                points2d_224=transform_points(result["points2d"], video["geometry"]),
                intrinsics_224=np.asarray(video["geometry"]["A"]) @ k,
                camera_poses_source=poses,
                content_mask=content_mask(video["geometry"]),
            )
        checksums.append(
            {
                "rgb": hashlib.sha256(value.tobytes()).hexdigest(),
                "points": hashlib.sha256(result["points2d_224"].tobytes()).hexdigest(),
                "K": hashlib.sha256(result["intrinsics_224"].tobytes()).hexdigest(),
            }
        )
        latencies.append(time.perf_counter() - tick)
    end = time.perf_counter()
    for m, c in containers.values():
        c.close()
        m.close()
    numeric.close()
    rgb.close()
    return {
        "init_seconds": init,
        "steady_start": steady_start,
        "steady_end": end,
        "latencies": latencies,
        "checksums": checksums,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("release", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    reader = JPEG224Reader(args.cache)
    operations = []
    rng = np.random.default_rng(20260915)
    numeric = MMapMotionReader(args.release / "subsets/molmospaces")
    for video in list(reader.videos)[:12]:
        _, v = reader.videos[video]
        n = numeric.get_object_full("molmospaces/object/" + video)["points2d"].shape[1]
        for _ in range(4):
            start = int(rng.integers(0, max(1, v["count"] - 7)))
            operations.append((video, list(range(start, min(start + 8, v["count"]))), list(range(min(32, n)))))
    reader.close()
    numeric.close()
    results = []
    for workers in [0, 1, 4]:
        for round_id in range(3):
            paired = {}
            for mode in ["raw", "jpeg85"] if round_id % 2 == 0 else ["jpeg85", "raw"]:
                count = max(workers, 1)
                specs = [(str(args.release), str(args.cache), mode, operations[i::count]) for i in range(count)]
                tick = time.perf_counter()
                if workers:
                    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                        batches = list(pool.map(partition, specs))
                else:
                    batches = [partition(specs[0])]
                wall = time.perf_counter() - tick
                latencies = [latency for b in batches for latency in b["latencies"]]
                steady = max(b["steady_end"] for b in batches) - min(b["steady_start"] for b in batches)
                result = {
                    "workers": workers,
                    "round": round_id,
                    "mode": mode,
                    "requests": len(operations),
                    "wall_seconds": wall,
                    "including_startup_requests_per_second": len(operations) / wall,
                    "steady_wall_seconds": steady,
                    "steady_requests_per_second": len(operations) / steady,
                    "init_seconds": [b["init_seconds"] for b in batches],
                    "p50_ms": float(np.percentile(latencies, 50) * 1000),
                    "p95_ms": float(np.percentile(latencies, 95) * 1000),
                    "max_worker_peak_rss_kib": max(b["peak_rss_kib"] for b in batches),
                    "checksums": [b["checksums"] for b in batches],
                }
                results.append(result)
                paired[mode] = result["checksums"]
                print(json.dumps({k: v for k, v in result.items() if k != "checksums"}), flush=True)
            for raw, jpeg in zip(paired["raw"], paired["jpeg85"]):
                assert [(x["points"], x["K"]) for x in raw] == [(x["points"], x["K"]) for x in jpeg]
    write_json(
        args.report,
        {
            "scope": "independent joint reader, NOT training DataLoader",
            "cache_state": "uncontrolled OS cache; no shared cache clearing",
            "raw_method": (
                "persistent bounded containers; keyframe seek then decode through requested "
                "contiguous window; Pillow224"
            ),
            "initialization_caveat": (
                "both paths load shared JPEG224 request metadata; startup includes process spawning and indices"
            ),
            "mmap": "numeric arrays and frame index only; JPEG payload uses persistent bounded seek/read handles",
            "lossy": "JPEG85 RGB is not pixel-equivalent to raw RGB224",
            "operations": operations,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
