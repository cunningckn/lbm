"""Bounded MP4 tar-member decoding and indexed image-cache pilot."""

from __future__ import annotations

import argparse
import io
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image

from .common import read_json, read_parquet_rows, write_checksum_manifest, write_json
from .delivery import audit_component, code_fingerprint


class MemberIO(io.RawIOBase):
    def __init__(self, path, offset, size):
        self.handle = open(path, "rb")
        self.offset, self.size, self.position = int(offset), int(size), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if position < 0 or position > self.size:
            raise ValueError("member seek out of bounds")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        self.handle.seek(self.offset + self.position)
        value = self.handle.read(size)
        self.position += len(value)
        return value

    def close(self):
        self.handle.close()
        super().close()


class RGBReader:
    def __init__(self, root):
        self.root = Path(root)
        audit_component(self.root)
        self.index = np.load(self.root / "frames.npy", mmap_mode="r")
        self.metadata = read_json(self.root / "dataset.json")
        if self.metadata.get("format_version") != 1 or self.metadata.get("state") != "rgb-pilot":
            raise ValueError("unsupported RGB cache schema/state")

    def get(self, frame_ids, codec="png"):
        if codec not in {"png", "jpeg"}:
            raise ValueError(codec)
        ids = np.asarray(frame_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(self.index)):
            raise IndexError("explicit frame IDs out of bounds")
        images = []
        with (self.root / f"{codec}.frames.bin").open("rb") as handle:
            for i in ids:
                row = self.index[int(i)]
                handle.seek(int(row[f"{codec}_offset"]))
                images.append(
                    np.asarray(Image.open(io.BytesIO(handle.read(int(row[f"{codec}_length"])))).convert("RGB"))
                )
        return {
            "rgb": np.stack(images),
            "frame_ids": ids,
            "pts": self.index["pts"][ids],
            "time_base": self.metadata["time_base"],
        }


def build_video(assets, row, clip, destination, alignment="strict-time"):
    contract = {
        "code": code_fingerprint(),
        "schema": 1,
        "clip": clip,
        "asset": row,
        "codecs": ["png", "jpeg-q95"],
        "size": "original",
        "alignment": alignment,
    }
    # The exact member, not its container pathname, defines the input content.
    import hashlib

    digest = hashlib.sha256()
    with MemberIO(assets / row["archive"], row["offset"], row["size"]) as member:
        while block := member.read(1024 * 1024):
            digest.update(block)
    contract["input_sha256"] = digest.hexdigest()
    if destination.exists():
        audit_component(destination, verify_files=True)
        if read_json(destination / "dataset.json")["contract"] != contract:
            raise ValueError("RGB resume contract mismatch")
        return read_json(destination / "metrics.json")
    staging = Path(tempfile.mkdtemp(prefix=".rgb-", dir=destination.parent))
    start = time.perf_counter()
    indices, mse, count = [], 0.0, 0
    previous_pts, time_base = None, None
    with (
        MemberIO(assets / row["archive"], row["offset"], row["size"]) as member,
        av.open(member) as container,
        (staging / "png.frames.bin").open("wb") as png,
        (staging / "jpeg.frames.bin").open("wb") as jpeg,
    ):
        for i, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or (previous_pts is not None and frame.pts <= previous_pts):
                raise ValueError("missing/non-monotonic PTS")
            previous_pts = frame.pts
            frame_time_base = [frame.time_base.numerator, frame.time_base.denominator]
            if time_base is not None and time_base != frame_time_base:
                raise ValueError("time base changed")
            time_base = frame_time_base
            if (frame.height, frame.width) != (clip["height"], clip["width"]):
                raise ValueError("source video dimensions disagree with annotations")
            rgb = frame.to_ndarray(format="rgb24")
            offsets = []
            for codec, stream in [("PNG", png), ("JPEG", jpeg)]:
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format=codec, **({"quality": 95} if codec == "JPEG" else {}))
                encoded = buffer.getvalue()
                decoded = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
                if codec == "PNG":
                    np.testing.assert_array_equal(rgb, decoded)
                else:
                    mse += float(np.square(rgb.astype(np.float64) - decoded).sum())
                    count += rgb.size
                offsets.extend([stream.tell(), len(encoded)])
                stream.write(encoded)
            indices.append((*offsets, i, frame.pts))
        if len(indices) != clip["num_frames"]:
            raise ValueError(f"frame count {len(indices)} != {clip['num_frames']}")
    index = np.array(
        indices,
        dtype=[
            ("png_offset", "<u8"),
            ("png_length", "<u8"),
            ("jpeg_offset", "<u8"),
            ("jpeg_length", "<u8"),
            ("frame_id", "<u8"),
            ("pts", "<i8"),
        ],
    )
    seconds = index["pts"].astype(float) * time_base[0] / time_base[1]
    expected = np.arange(len(index)) / clip["fps"]
    aligned = bool(np.allclose(seconds - seconds[0], expected, atol=1e-5))
    if not aligned and alignment == "strict-time":
        diagnostic = {
            "state": "blocked-timestamp-alignment",
            "video_id": clip["video_id"],
            "frames": len(index),
            "annotation_fps": clip["fps"],
            "time_base": time_base,
            "pts_first": index["pts"][:10].tolist(),
            "actual_duration": float(seconds[-1] - seconds[0]),
            "annotation_duration": float(expected[-1]),
            "max_error_seconds": float(np.max(np.abs(seconds - seconds[0] - expected))),
            "png_bytes": (staging / "png.frames.bin").stat().st_size,
            "jpeg_bytes": (staging / "jpeg.frames.bin").stat().st_size,
            "png_exact": True,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        write_json(staging / "ALIGNMENT_FAILURE.json", diagnostic)
        raise ValueError(json.dumps(diagnostic))
    np.save(staging / "frames.npy", index)
    metrics = {
        "video_id": clip["video_id"],
        "frames": len(index),
        "source_bytes": row["size"],
        "png_bytes": (staging / "png.frames.bin").stat().st_size,
        "jpeg_bytes": (staging / "jpeg.frames.bin").stat().st_size,
        "jpeg_psnr_db": float(10 * np.log10(255**2 / (mse / count))),
        "png_exact": True,
        "pts_aligned": aligned,
        "alignment": alignment,
        "max_time_error_seconds": float(np.max(np.abs(seconds - seconds[0] - expected))),
        "elapsed_seconds": time.perf_counter() - start,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(
        staging / "dataset.json",
        {
            "format_version": 1,
            "contract": contract,
            "time_base": time_base,
            "state": "rgb-pilot",
            "frame_count": len(index),
            "annotation_fps": clip["fps"],
            "alignment": alignment,
            "time_semantics_verified": aligned,
        },
    )
    write_json(staging / "metrics.json", metrics)
    write_json(
        staging / "PILOT_READY.json",
        {"format_version": 1, "status": "pilot-ready", "sha256sums_sha256": write_checksum_manifest(staging)},
    )
    os.rename(staging, destination)
    reader = RGBReader(destination)
    assert reader.get([0, len(index) - 1])["rgb"].shape[0] == 2
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alignment", choices=["strict-time", "frame-index"], default="strict-time")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.release / "assets"
    rows = {r["member_name"]: r for r in read_parquet_rows(assets / "assets_index.parquet")}
    clips = read_parquet_rows(args.release / "subsets/molmospaces/clips.parquet")
    selected, keys = [], set()
    for clip in clips:
        key = (clip["num_frames"], clip["video_id"].rsplit("__", 1)[-1])
        if key in keys:
            continue
        selected.append(clip)
        keys.add(key)
        if len(selected) == 4:
            break
    results = []
    for clip in selected:
        name = "videos/" + clip["video_id"] + ".mp4"
        if name not in rows:
            raise KeyError(name)
        result = build_video(assets, rows[name], clip, args.output / clip["video_id"], args.alignment)
        results.append(result)
        print(json.dumps(result), flush=True)
    write_json(args.output / "RGB_PILOT_RESULTS.json", results)


if __name__ == "__main__":
    main()
