"""Matched q85/q95 pilot comparison, with alternating benchmark order."""

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
from molmo_motion_cache.common import read_json, sha256_file, write_json
from molmo_motion_cache.delivery import audit_component
from molmo_motion_cache.rgb_benchmark import run_batch
from molmo_motion_cache.rgb_pilot import RGBReader
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("q85", type=Path)
    parser.add_argument("q95", type=Path)
    args = parser.parse_args()
    names = sorted(p.parent.name for p in args.q85.glob("*/PILOT_READY.json"))
    assert names and names == sorted(p.parent.name for p in args.q95.glob("*/PILOT_READY.json"))
    checks, panels = [], []
    for name in names:
        low, high = args.q85 / name, args.q95 / name
        audit_component(low, verify_files=True)
        audit_component(high, verify_files=True)
        a, b = RGBReader(low), RGBReader(high)
        assert a.metadata["contract"]["input_sha256"] == b.metadata["contract"]["input_sha256"]
        assert a.metadata["time_base"] == b.metadata["time_base"]
        for field in ("frame_id", "pts"):
            np.testing.assert_array_equal(a.index[field], b.index[field])
        assert sha256_file(low / "png.frames.bin") == sha256_file(high / "png.frames.bin")
        # Choose the middle frame in each view; show an explicitly central 192px crop.
        frame = len(a.index) // 2
        images = [a.get([frame], "png")["rgb"][0], b.get([frame], "jpeg")["rgb"][0], a.get([frame], "jpeg")["rgb"][0]]
        crops = []
        for pixels in images:
            h, w = pixels.shape[:2]
            crop = Image.fromarray(pixels).crop((w // 2 - 96, h // 2 - 96, w // 2 + 96, h // 2 + 96))
            crops.append(crop.resize((384, 384), Image.Resampling.NEAREST))
        panels.append((name, frame, crops))
        m85, m95 = read_json(low / "metrics.json"), read_json(high / "metrics.json")
        checks.append(
            {
                "video": name,
                "frames": len(a.index),
                "q85": m85,
                "q95": m95,
                "same_source_pixels": True,
                "same_frame_ids_and_pts": True,
            }
        )
    canvas = Image.new("RGB", (1152, len(panels) * 420), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (name, frame, crops) in enumerate(panels):
        draw.text((5, row * 420), f"{name} | frame {frame} | center crop, 2x nearest", fill="black")
        for col, (label, crop) in enumerate(zip(["Decoded source PNG", "JPEG q95", "JPEG q85"], crops)):
            draw.text((col * 384 + 5, row * 420 + 16), label, fill="black")
            canvas.paste(crop, (col * 384, row * 420 + 36))
    canvas.save(args.q85 / "JPEG_COMPARISON.png")
    timings = []
    for workers in (0, 1, 4):
        for repeat in range(3):
            for label, root in (
                [("q85", args.q85), ("q95", args.q95)] if repeat % 2 == 0 else [("q95", args.q95), ("q85", args.q85)]
            ):
                specs = []
                for name in names:
                    n = read_json(root / name / "dataset.json")["frame_count"]
                    rng = np.random.default_rng(20260915)
                    ops = [sorted(rng.choice(n, min(8, n), replace=False).tolist()) for _ in range(12)]
                    specs.append(("jpeg", str(root / name), "unused", ops))
                tick = time.perf_counter()
                if workers:
                    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                        results = list(pool.map(run_batch, specs))
                else:
                    results = [run_batch(spec) for spec in specs]
                elapsed = time.perf_counter() - tick
                timings.append(
                    {
                        "workers": workers,
                        "repeat": repeat,
                        "quality": label,
                        "requests": 48,
                        "requests_per_second_including_startup": 48 / elapsed,
                        "p95_request_seconds": float(np.percentile([v for r in results for v in r["latencies"]], 95)),
                    }
                )
    summary = []
    for workers in (0, 1, 4):
        for label in ("q85", "q95"):
            values = [
                v["requests_per_second_including_startup"]
                for v in timings
                if v["workers"] == workers and v["quality"] == label
            ]
            summary.append(
                {"workers": workers, "quality": label, "median_requests_per_second": float(np.median(values))}
            )
    report = {
        "scope": "same four videos, 1064 frames, original size, same Pillow subsampling defaults",
        "checks": checks,
        "timings": timings,
        "summary": summary,
        "cache_state": "uncontrolled OS cache, 3 alternating trials, no cold-cache or training-performance claim",
    }
    write_json(args.q85 / "JPEG_QUALITY_COMPARISON.json", report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
