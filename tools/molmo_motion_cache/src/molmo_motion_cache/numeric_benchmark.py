"""DROID matched numerical windows with 0/1/4 subprocess workers."""

import argparse
import hashlib
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import write_json
from .reader import MMapDroidReader, RawTarDroidReader

_STATE = None


def batch(spec):
    global _STATE
    source, cache, mode, seed = spec
    tick = time.perf_counter()
    identity = (source, cache, mode)
    if _STATE is None or _STATE[0] != identity:
        cached = MMapDroidReader(cache)
        reader = RawTarDroidReader(source, cached) if mode == "raw" else cached
        _STATE = (identity, cached, reader, cached.sample_ids)
    _, cached, reader, sample_ids = _STATE
    initialized = time.perf_counter() - tick
    rng = np.random.default_rng(seed)
    latencies, digests = [], []
    for _ in range(32):
        sample = sample_ids[int(rng.integers(0, min(512, len(sample_ids))))]
        start = int(rng.integers(0, max(1, cached.tracks[sample]["num_frames"] - 7)))
        tick = time.perf_counter()
        window = reader.get_window(sample, start=start, frames=8, points=32)
        digest = hashlib.sha256()
        for name, array in sorted(window.items()):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        digests.append(digest.hexdigest())
        latencies.append(time.perf_counter() - tick)
    return {
        "initialization_seconds": initialized,
        "first_request_seconds": latencies[0],
        "steady_request_seconds": latencies[1:],
        "digests": digests,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    results = []
    for workers in (0, 1, 4):
        for mode in ("raw", "cache"):
            specs = [(str(args.source), str(args.cache), mode, 20260915 + i) for i in range(4)]
            tick = time.perf_counter()
            if workers:
                with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    rows = list(pool.map(batch, specs))
            else:
                rows = [batch(spec) for spec in specs]
            elapsed = time.perf_counter() - tick
            steady = [v for row in rows for v in row["steady_request_seconds"]]
            results.append(
                {
                    "workers": workers,
                    "mode": mode,
                    "wall_seconds_including_startup": elapsed,
                    "requests_per_second_including_startup": 128 / elapsed,
                    "steady_latency_p50_seconds": float(np.median(steady)),
                    "steady_latency_p95_seconds": float(np.percentile(steady, 95)),
                    "batches": rows,
                }
            )
            print(workers, mode, 128 / elapsed, flush=True)
        assert [r["digests"] for r in results[-1]["batches"]] == [r["digests"] for r in results[-2]["batches"]]
    write_json(
        args.report,
        {
            "scope": "DROID numerical 8 frames x 32 points; 4 batches x 32 requests",
            "cache_state": "OS cache uncontrolled; first request is not system cold cache",
            "checksum_exact": True,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
