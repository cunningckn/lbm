"""Experimental JPEG224 sharded cache: bounded decode, immutable publication."""

import argparse
import hashlib
import io
import os
import resource
import tempfile
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import PIL
from PIL import Image, features

from .common import (
    checksum_entries,
    read_json,
    read_parquet_rows,
    safe_relative_path,
    sha256_file,
    write_checksum_manifest,
    write_json,
)
from .delivery import audit_component, code_fingerprint
from .geometry224 import REFERENCE, resize224
from .rgb_pilot import MemberIO

FRAME_DTYPE = np.dtype([("shard", "<u8"), ("offset", "<u8"), ("length", "<u8"), ("frame_id", "<u8"), ("pts", "<i8")])


def dependencies():
    return {
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "pyav": av.__version__,
        "libjpeg": features.version_codec("jpg"),
        "ffmpeg": {k: list(v) for k, v in av.library_versions.items()},
    }


def encode(rgb, quality):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, "JPEG", quality=quality, subsampling=2, optimize=False, progressive=False)
    return buffer.getvalue()


def member_hash(assets, row):
    digest = hashlib.sha256()
    with MemberIO(assets / safe_relative_path(row["archive"]), row["offset"], row["size"]) as member:
        while block := member.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sample_clips(clips, count=100, seed=20260915):
    """Retain historical first four, then seeded round-robin strata."""
    unique = {c["video_id"]: c for c in clips}
    selected, keys = [], set()
    for c in unique.values():
        key = (c["num_frames"], c["video_id"].rsplit("__", 1)[-1])
        if key not in keys:
            selected.append(c)
            keys.add(key)
        if len(selected) == min(4, count):
            break
    groups = defaultdict(list)
    chosen = {c["video_id"] for c in selected}
    lengths = np.quantile([c["num_frames"] for c in unique.values()], [0.25, 0.5, 0.75])
    for c in unique.values():
        if c["video_id"] in chosen:
            continue
        parts = c["video_id"].split("__")
        key = (parts[0], parts[1], parts[-1], int(np.searchsorted(lengths, c["num_frames"])), c["split"])
        groups[key].append(c)
    rng = np.random.default_rng(seed)
    buckets = [sorted(groups[k], key=lambda c: c["video_id"]) for k in sorted(groups)]
    rng.shuffle(buckets)
    for b in buckets:
        rng.shuffle(b)
    while len(selected) < min(count, len(unique)):
        for b in buckets:
            if b:
                selected.append(b.pop())
                if len(selected) == min(count, len(unique)):
                    break
    return selected


def publish(root):
    digest = write_checksum_manifest(root)
    write_json(root / "PILOT_READY.json", {"format_version": 1, "status": "pilot-ready", "sha256sums_sha256": digest})
    audit_component(root, verify_files=True)


def build_group(spec):
    assets, destination, clips, config = spec
    assets, destination = Path(assets), Path(destination)
    contract = {"config": config, "videos": clips}
    if destination.exists():
        audit_component(destination, verify_files=True)
        if read_json(destination / "dataset.json")["contract"] != contract:
            raise ValueError("JPEG224 resume contract mismatch")
        return str(destination)
    staging = Path(tempfile.mkdtemp(prefix=".jpeg224-", dir=destination.parent))
    total = sum(c["clip"]["num_frames"] for c in clips)
    index = np.lib.format.open_memmap(staging / "frames.npy", mode="w+", dtype=FRAME_DTYPE, shape=(total,))
    shard, handle, cursor = -1, None, 0
    videos = {}
    started = time.perf_counter()
    try:
        for item in clips:
            clip, row = item["clip"], item["asset"]
            begin, prev, tb, geometry = cursor, None, None, None
            with (
                MemberIO(assets / safe_relative_path(row["archive"]), row["offset"], row["size"]) as member,
                av.open(member) as container,
            ):
                container.streams.video[0].thread_count = 1
                for fid, frame in enumerate(container.decode(video=0)):
                    if fid >= clip["num_frames"]:
                        raise ValueError("too many video frames")
                    current_tb = [frame.time_base.numerator, frame.time_base.denominator]
                    if (
                        frame.pts is None
                        or (prev is not None and frame.pts <= prev)
                        or (tb is not None and tb != current_tb)
                    ):
                        raise ValueError("invalid PTS/time base")
                    prev, tb = frame.pts, current_tb
                    rgb, geom = resize224(
                        frame.to_ndarray(format="rgb24"),
                        source_size=[clip["width"], clip["height"]],
                        convention=config["coordinate_convention"],
                    )
                    if geometry is not None and geom != geometry:
                        raise ValueError("geometry changed within video")
                    geometry = geom
                    payload = encode(rgb, config["quality"])
                    if handle is None or (handle.tell() and handle.tell() + len(payload) > config["shard_bytes"]):
                        if handle:
                            handle.flush()
                            os.fsync(handle.fileno())
                            handle.close()
                        shard += 1
                        handle = (staging / f"frames-{shard:05d}.bin").open("xb")
                    index[cursor] = (shard, handle.tell(), len(payload), fid, frame.pts)
                    handle.write(payload)
                    cursor += 1
            if cursor - begin != clip["num_frames"]:
                raise ValueError("video/annotation frame count mismatch")
            if member_hash(assets, row) != item["input_sha256"]:
                raise ValueError("source changed during conversion")
            seconds = index["pts"][begin:cursor].astype(float) * tb[0] / tb[1]
            error = float(np.max(np.abs(seconds - seconds[0] - np.arange(cursor - begin) / clip["fps"])))
            videos[clip["video_id"]] = {
                **item,
                "start": begin,
                "count": cursor - begin,
                "geometry": geometry,
                "time_base": tb,
                "max_clock_error_seconds": error,
            }
            print(f"jpeg224 q{config['quality']} {len(videos)}/{len(clips)} {clip['video_id']}", flush=True)
    finally:
        if handle:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        index.flush()
        del index
    write_json(
        staging / "dataset.json",
        {
            "format_version": 1,
            "state": "jpeg224-frame-index-pilot",
            "contract": contract,
            "videos": videos,
            "time_semantics_verified": False,
            "source_coordinate_convention_verified": False,
        },
    )
    write_json(
        staging / "metrics.json",
        {
            "frames": cursor,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    publish(staging)
    os.rename(staging, destination)
    return str(destination)


def build(
    release,
    destination,
    *,
    count=100,
    quality=85,
    workers=4,
    shard_bytes=2 * 1024**3,
    convention="integer-center",
    seed=20260915,
):
    if quality not in {85, 95} or not 1 <= workers <= 4 or count < 1 or shard_bytes < 1024:
        raise ValueError("invalid bounded experimental configuration")
    if convention not in {"integer-center", "pixel-boundary"}:
        raise ValueError("unknown coordinate convention")
    clips = read_parquet_rows(release / "subsets/molmospaces/clips.parquet")
    selected = sample_clips(clips, count, seed)
    rows = {r["member_name"]: r for r in read_parquet_rows(release / "assets/assets_index.parquet")}
    inputs = []
    for c in selected:
        row = rows["videos/" + c["video_id"] + ".mp4"]
        inputs.append({"clip": c, "asset": row, "input_sha256": member_hash(release / "assets", row)})
    config = {
        "schema": "jpeg224-v1",
        "code": code_fingerprint(),
        "reference": REFERENCE,
        "dependencies": dependencies(),
        "quality": quality,
        "subsampling": 2,
        "optimize": False,
        "progressive": False,
        "target_size": [224, 224],
        "resize": "Pillow.ImageOps.contain/BILINEAR/center/black",
        "coordinate_convention": convention,
        "alignment": "explicit-frame-index-unverified",
        "shard_bytes": shard_bytes,
        "workers": workers,
        "seed": seed,
    }
    contract = {"config": config, "inputs": inputs}
    if destination.exists():
        audit_component(destination, verify_files=True)
        if read_json(destination / "dataset.json")["contract"] != contract:
            raise ValueError("JPEG224 root resume contract mismatch")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rgb224-run-", dir=destination.parent))
    started = time.perf_counter()
    specs = [
        (str(release / "assets"), str(staging / f"group-{i:03d}"), inputs[i::workers], config)
        for i in range(min(workers, len(inputs)))
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(build_group, specs))
    write_json(
        staging / "dataset.json",
        {
            "format_version": 1,
            "state": "jpeg224-frame-index-pilot",
            "contract": contract,
            "groups": [f"group-{i:03d}" for i in range(len(specs))],
            "known_unique_videos": len({c["video_id"] for c in clips}),
            "known_unique_frames": sum(c["num_frames"] for c in {c["video_id"]: c for c in clips}.values()),
            "time_semantics_verified": False,
            "source_coordinate_convention_verified": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    publish(staging)
    os.rename(staging, destination)
    return destination


class JPEG224Reader:
    """One instance per worker, mmap indices and bounded persistent file handles."""

    def __init__(self, root, max_open_files=16):
        self.root = Path(root)
        audit_component(self.root)
        self.metadata = read_json(self.root / "dataset.json")
        if self.metadata.get("format_version") != 1 or self.metadata.get("state") != "jpeg224-frame-index-pilot":
            raise ValueError("unsupported JPEG224 schema")
        self.videos, self.indices = {}, {}
        self.handles = OrderedDict()
        self.max_open_files = max_open_files
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        manifest = checksum_entries(self.root)
        for group in self.metadata["groups"]:
            safe_relative_path(group)
            doc = read_json(self.root / group / "dataset.json")
            if manifest.get(f"{group}/frames.npy") != sha256_file(self.root / group / "frames.npy"):
                raise ValueError("frame index checksum mismatch")
            index = np.load(self.root / group / "frames.npy", mmap_mode="r", allow_pickle=False)
            if index.dtype != FRAME_DTYPE:
                raise ValueError("invalid frame index dtype")
            self.indices[group] = index
            for video, row in doc["videos"].items():
                if video in self.videos or row["start"] < 0 or row["start"] + row["count"] > len(index):
                    raise ValueError("duplicate video or invalid frame range")
                self.videos[video] = (group, row)

    def get(self, video_id, frame_ids):
        group, video = self.videos[video_id]
        ids = np.asarray(frame_ids)
        if (
            ids.ndim != 1
            or not len(ids)
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= video["count"])
        ):
            raise IndexError("invalid explicit frame IDs")
        rows = self.indices[group][video["start"] + ids.astype(np.int64)]
        if not np.array_equal(rows["frame_id"], ids):
            raise ValueError("frame index identity mismatch")
        images = []
        for row in rows:
            key = (group, int(row["shard"]))
            handle = self.handles.pop(key, None)
            if handle is None:
                handle = (self.root / group / f"frames-{key[1]:05d}.bin").open("rb")
            self.handles[key] = handle
            while len(self.handles) > self.max_open_files:
                self.handles.popitem(last=False)[1].close()
            offset, length = int(row["offset"]), int(row["length"])
            if offset + length > os.fstat(handle.fileno()).st_size or length < 1:
                raise ValueError("invalid frame byte range")
            handle.seek(offset)
            payload = handle.read(length)
            with Image.open(io.BytesIO(payload)) as image:
                if image.size != (224, 224) or image.format != "JPEG":
                    raise ValueError("invalid JPEG224 payload")
                images.append(np.asarray(image.convert("RGB")))
        return {
            "rgb": np.stack(images),
            "frame_ids": ids.copy(),
            "pts": rows["pts"].copy(),
            "time_base": video["time_base"],
            "geometry": video["geometry"],
            "time_semantics_verified": False,
            "source_coordinate_convention_verified": False,
        }

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        for index in self.indices.values():
            index._mmap.close()
        self.indices.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--quality", type=int, choices=[85, 95], default=85)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--coordinate-convention", choices=["integer-center", "pixel-boundary"], required=True)
    parser.add_argument("--frame-index-experiment", action="store_true", required=True)
    args = parser.parse_args()
    build(
        args.release,
        args.output,
        count=args.count,
        quality=args.quality,
        workers=args.workers,
        shard_bytes=args.shard_bytes,
        convention=args.coordinate_convention,
    )


if __name__ == "__main__":
    main()
