"""Read-only, paired RGB224 quality/geometry evidence; reports live outside cache."""

import argparse
import ast
import hashlib
import io
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import av
import numpy as np
from molmo_motion_cache.common import write_json
from molmo_motion_cache.delivery import audit_component
from molmo_motion_cache.generic_reader import MMapMotionReader
from molmo_motion_cache.geometry224 import REFERENCE, resize224
from molmo_motion_cache.rgb224 import JPEG224Reader, encode
from molmo_motion_cache.rgb_pilot import MemberIO
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity


def load_reference(path):
    text = path.read_bytes()
    if hashlib.sha256(text).hexdigest() != REFERENCE["sha256"]:
        raise ValueError("reference file changed")
    tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resize_with_pad")
    namespace = {"np": np, "Image": Image, "ImageOps": ImageOps, "_RESAMPLE_BILINEAR": Image.Resampling.BILINEAR}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<frozen-reference>", "exec"), namespace)
    return namespace["resize_with_pad"]


def evaluate(release, cache, report, reference_path):
    if report.exists():
        raise FileExistsError(report)
    report.mkdir(parents=True)
    audit = audit_component(cache, verify_files=True)
    reference = load_reference(reference_path)
    reader = JPEG224Reader(cache)
    numeric = MMapMotionReader(release / "subsets/molmospaces")
    result, sizes, sheets = [], {85: [], 95: []}, []
    started = time.perf_counter()
    for video_id, (_, video) in reader.videos.items():
        row, clip = video["asset"], video["clip"]
        left, top = video["geometry"]["padding"][:2]
        w, h = video["geometry"]["content_size"]
        probes = {0, video["count"] // 2, video["count"] - 1}
        error = {85: 0.0, 95: 0.0}
        ssim = {85: [], 95: []}
        selected_hashes, pixel_count = {}, 0
        with (
            MemberIO(release / "assets" / row["archive"], row["offset"], row["size"]) as member,
            av.open(member) as container,
        ):
            container.streams.video[0].thread_count = 1
            for fid, frame in enumerate(container.decode(video=0)):
                source = frame.to_ndarray(format="rgb24")
                rgb, g = resize224(source, source_size=[clip["width"], clip["height"]])
                if g != video["geometry"]:
                    raise AssertionError("geometry differs")
                content = rgb[top : top + h, left : left + w]
                decoded = {}
                for q in [85, 95]:
                    data = encode(rgb, q)
                    sizes[q].append(len(data))
                    with Image.open(io.BytesIO(data)) as im:
                        decoded[q] = np.asarray(im.convert("RGB"))
                    diff = content.astype(float) - decoded[q][top : top + h, left : left + w]
                    error[q] += float(np.square(diff).sum())
                    if fid in probes:
                        ssim[q].append(
                            float(
                                structural_similarity(
                                    content, decoded[q][top : top + h, left : left + w], channel_axis=-1, data_range=255
                                )
                            )
                        )
                pixel_count += content.size
                if fid in probes:
                    np.testing.assert_array_equal(rgb, reference(source, 224, 0))
                    np.testing.assert_array_equal(reader.get(video_id, [fid])["rgb"][0], decoded[85])
                    buffer = io.BytesIO()
                    Image.fromarray(rgb).save(buffer, "PNG")
                    np.testing.assert_array_equal(rgb, np.asarray(Image.open(io.BytesIO(buffer.getvalue()))))
                    selected_hashes[str(fid)] = hashlib.sha256(rgb.tobytes()).hexdigest()
                    if len(result) < 4:
                        if fid == video["count"] // 2:
                            sheets.append(np.concatenate([rgb, decoded[95], decoded[85]], axis=1))
                        Image.fromarray(rgb).save(report / f"reference-{len(result):02d}-{fid:04d}.png")
        sample = "molmospaces/object/" + video_id
        ids = sorted(probes)
        full = numeric.get_object_full(sample)
        pts = list(range(min(8, full["points2d"].shape[1])))
        joined = numeric.get_selection(sample, frame_ids=ids, point_ids=pts, jpeg224_root=cache, require_rgb=True)
        np.testing.assert_array_equal(joined["points3d"], full["points3d"][np.ix_(ids, pts)])
        np.testing.assert_array_equal(joined["visibility2d"], full["visibility2d"][np.ix_(ids, pts)])
        sx, sy = w / clip["width"], h / clip["height"]
        expected = (joined["points2d_source"] + 0.5) * [sx, sy] - 0.5 + [left, top]
        np.testing.assert_allclose(joined["points2d_224"], expected, atol=1e-5, equal_nan=True)
        np.testing.assert_allclose(joined["intrinsics_224"], np.asarray(g["A"]) @ joined["intrinsics_source_pixels"])
        # Inspect released 3D->2D consistency without interpreting residual as resize error.
        xyz = joined["points3d"].astype(float)
        c2w = joined["camera_poses_source"].astype(float)
        xyzh = np.concatenate([xyz, np.ones((*xyz.shape[:-1], 1))], axis=-1)
        cam = np.einsum("tij,tpj->tpi", np.linalg.inv(c2w), xyzh)[..., :3]
        uvh = np.einsum("tij,tpj->tpi", joined["intrinsics_source_pixels"], cam)
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = uvh[..., :2] / uvh[..., 2:]
        valid = (
            joined["visibility2d"]
            & joined["visibility3d"]
            & (cam[..., 2] > 0)
            & np.isfinite(uv).all(-1)
            & np.isfinite(joined["points2d_source"]).all(-1)
        )
        residual = np.linalg.norm(uv - joined["points2d_source"], axis=-1)[valid]
        record = {
            "video_id": video_id,
            "frames": video["count"],
            "source_bytes": row["size"],
            "content_psnr_db": {q: float(10 * np.log10(255**2 / (error[q] / pixel_count))) for q in error},
            "content_ssim_first_middle_last": ssim,
            "reference_hashes": selected_hashes,
            "joint_selection_passed": True,
            "source_projection_samples": len(residual),
            "source_projection_median_px": float(np.median(residual)) if len(residual) else None,
            "clock_error_seconds": video["max_clock_error_seconds"],
        }
        result.append(record)
        print(f"evaluated {len(result)}/{len(reader.videos)} {video_id}", flush=True)
    if sheets:
        Image.fromarray(np.concatenate(sheets, axis=0)).save(report / "JPEG224_COMPARISON.png")
    files = [p for p in cache.rglob("*") if p.is_file()]
    summary = {
        "scope": "MolmoSpaces JPEG224 frame-index experiment; not training benchmark",
        "full_hash_audit": audit,
        "reference": REFERENCE,
        "videos": len(result),
        "frames": len(sizes[85]),
        "mp4_bytes": sum(r["source_bytes"] for r in result),
        "jpeg_bytes": {q: sum(sizes[q]) for q in sizes},
        "jpeg_frame_bytes_percentiles": {
            q: dict(zip(["min", "p50", "p95", "max"], np.percentile(sizes[q], [0, 50, 95, 100]).tolist()))
            for q in sizes
        },
        "cache_logical_bytes": sum(p.stat().st_size for p in files),
        "cache_allocated_bytes": sum(p.stat().st_blocks * 512 for p in files),
        "known_unique_frames": reader.metadata["known_unique_frames"],
        "known_unique_videos": reader.metadata["known_unique_videos"],
        "estimated_full_jpeg_bytes": {
            q: sum(sizes[q]) / len(sizes[q]) * reader.metadata["known_unique_frames"] for q in sizes
        },
        "estimate_caveat": "stratified pilot mean, not population-weighted; not guaranteed capacity",
        "build_wall_seconds": reader.metadata["elapsed_seconds"],
        "evaluation_wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "time_semantics_verified": False,
        "source_coordinate_convention_verified": False,
        "q95_payload_retained": False,
        "png_reference_count": len(list(report.glob("reference*.png"))),
        "results": result,
    }
    write_json(report / "QUALITY_GEOMETRY.json", summary)
    numeric.close()
    reader.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("release", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("reference", type=Path)
    a = p.parse_args()
    evaluate(a.release, a.cache, a.report, a.reference)
