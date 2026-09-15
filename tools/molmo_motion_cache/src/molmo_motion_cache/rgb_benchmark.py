"""Paired RGB frame requests with parent/worker initialization separated."""

import argparse
import hashlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np

from .common import read_json, write_json
from .rgb_pilot import MemberIO, RGBReader


def run_batch(spec):
    mode, cache, assets, operations = spec
    tick = time.perf_counter()
    reader = RGBReader(cache)
    contract = reader.metadata["contract"]
    row = contract["asset"]
    initialization = time.perf_counter() - tick
    latencies, checksums = [], []
    started = time.perf_counter()
    for frame_ids in operations:
        tick = time.perf_counter()
        if mode == "raw":
            frames = {}
            with (
                MemberIO(Path(assets) / row["archive"], row["offset"], row["size"]) as member,
                av.open(member) as container,
            ):
                for index, frame in enumerate(container.decode(video=0)):
                    if index in frame_ids:
                        frames[index] = frame.to_ndarray(format="rgb24")
                    if index >= max(frame_ids):
                        break
            value = np.stack([frames[i] for i in frame_ids])
        else:
            value = reader.get(frame_ids, codec=mode)["rgb"]
        checksums.append(hashlib.sha256(value.tobytes()).hexdigest())
        latencies.append(time.perf_counter() - tick)
    return {
        "init_seconds": initialization,
        "steady_seconds": time.perf_counter() - started,
        "latencies": latencies,
        "checksums": checksums,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot", type=Path)
    parser.add_argument("assets", type=Path)
    args = parser.parse_args()
    videos = sorted(p.parent for p in args.pilot.glob("*/PILOT_READY.json"))
    results = []
    for workers in [0, 1, 4]:
        for mode in ["raw", "png", "jpeg"]:
            specs = []
            for video in videos:
                n = read_json(video / "dataset.json")["frame_count"]
                rng = np.random.default_rng(20260915)
                operations = [sorted(rng.choice(n, min(8, n), replace=False).tolist()) for _ in range(12)]
                specs.append((mode, str(video), str(args.assets), operations))
            tick = time.perf_counter()
            if workers:
                with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    batches = list(pool.map(run_batch, specs))
            else:
                batches = [run_batch(spec) for spec in specs]
            elapsed = time.perf_counter() - tick
            latencies = [v for b in batches for v in b["latencies"]]
            result = {
                "workers": workers,
                "mode": mode,
                "requests": len(latencies),
                "wall_seconds_including_startup": elapsed,
                "requests_per_second_including_startup": len(latencies) / elapsed,
                "worker_initialization_seconds": [b["init_seconds"] for b in batches],
                "worker_steady_seconds": [b["steady_seconds"] for b in batches],
                "latency_p50_seconds": float(np.median(latencies)),
                "latency_p95_seconds": float(np.percentile(latencies, 95)),
                "max_worker_peak_rss_kib": max(b["peak_rss_kib"] for b in batches),
                "checksums": [b["checksums"] for b in batches],
            }
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "checksums"}), flush=True)
        assert results[-3]["checksums"] == results[-2]["checksums"], "raw/PNG pixel mismatch"
    write_json(
        args.pilot / "RGB_BENCHMARK.json",
        {
            "state": "rgb-pilot",
            "cache_state": "uncontrolled OS page cache; no cold-cache claim; no cache dropping",
            "scope": ("4 videos, 12 requests per video, 8 explicit random frames; "
                      "JPEG is lossy and not parity-equivalent"),
            "raw_png_exact": True,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
