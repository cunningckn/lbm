#!/usr/bin/env python3
"""Microbench JPEG decode for one training batch (no DataLoader).

Mirrors history=18, 3 cameras, batch=4 → 216 JPEGs, letterbox 224.
Compares serial vs thread-pool, and per-JPEG alloc vs chunked decode-into-dest.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from lbm.dataloader.mmap.frame_mmap_io import FRAMES_SUBDIR, MmapFrameEpisode
from lbm.dataloader.mmap.jpeg_io import decode_jpeg_rgb, decode_jpegs_into, decode_thread_init
from lbm.dataloader.mmap.mmap_io import MMAP_DIRNAME
from lbm.dataloader.paths import datasets_root

CACHE = datasets_root() / "rmbench" / MMAP_DIRNAME / FRAMES_SUBDIR
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
N_FRAMES = 18
BATCH = 4
REPEAT = 30


def _alloc_decode(blobs: list, executor: ThreadPoolExecutor | None) -> None:
    """One map task per JPEG, allocate a new array, no dest."""
    if executor is None:
        for blob in blobs:
            decode_jpeg_rgb(blob)
        return
    list(executor.map(decode_jpeg_rgb, blobs))


def main() -> None:
    episodes = [MmapFrameEpisode.open(CACHE / "episode_000000" / cam) for cam in CAMS]
    idx = np.arange(N_FRAMES)
    one: list = []
    for ep in episodes:
        one.extend(ep.gather_blobs(idx))
    blobs = one * BATCH
    n = len(blobs)
    h, w = episodes[0].height, episodes[0].width
    print(f"jpegs={n}  hw={h}x{w}  cams={len(CAMS)}  frames={N_FRAMES}  batch={BATCH}", flush=True)

    def _run(label: str, fn) -> None:
        fn()
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            fn()
        elapsed = time.perf_counter() - t0
        per = elapsed / REPEAT
        samp_s = BATCH / per
        print(
            f"{label:22s}  {per * 1e3:7.2f} ms/batch  "
            f"{n / per:7.0f} jpeg/s  {samp_s:7.1f} samp/s-eq",
            flush=True,
        )

    out = np.empty((n, h, w, 3), dtype=np.uint8)
    _run("into serial", lambda: decode_jpegs_into(blobs, out, None))
    _run("alloc serial", lambda: _alloc_decode(blobs, None))
    for threads in (2, 4, 8, 16):
        pool = ThreadPoolExecutor(max_workers=threads, initializer=decode_thread_init)
        try:
            _run(f"into pool-{threads}", lambda p=pool: decode_jpegs_into(blobs, out, p))
            _run(f"alloc pool-{threads}", lambda p=pool: _alloc_decode(blobs, p))
        finally:
            pool.shutdown(wait=False)
    for ep in episodes:
        ep.close()


if __name__ == "__main__":
    main()
