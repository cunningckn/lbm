"""PI0-compatible Pillow geometry; coordinate convention is an explicit input."""

import numpy as np
from PIL import Image, ImageOps

REFERENCE = {
    "repository_head": "9d75617c630efc71aa7dddebc81987e3ae8689b0",
    "file": "data/utils/image_preprocess.py",
    "sha256": "85c66cd60814b8169c85a2fae52026cfd80f89942fd8bb1ed3b0b08f594a5396",
    "function": "resize_with_pad",
}


def resize224(rgb, *, source_size=None, convention="integer-center"):
    """Return pixels and geometry. Merely choosing a convention does not verify it."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected RGB uint8 HWC")
    height, width = rgb.shape[:2]
    if min(height, width) <= 0 or (source_size is not None and list(source_size) != [width, height]):
        raise ValueError("source image/annotation dimensions disagree")
    if convention not in {"integer-center", "pixel-boundary"}:
        raise ValueError("coordinate convention must be explicit")
    fitted = ImageOps.contain(Image.fromarray(rgb), (224, 224), method=Image.Resampling.BILINEAR)
    w, h = fitted.size
    left, top = (224 - w) // 2, (224 - h) // 2
    canvas = Image.new("RGB", (224, 224), color=0)
    canvas.paste(fitted, (left, top))
    sx, sy = w / width, h / height
    tx, ty = left, top
    if convention == "integer-center":
        tx += (sx - 1) / 2
        ty += (sy - 1) / 2
    geometry = {
        "source_size": [width, height],
        "content_size": [w, h],
        "target_size": [224, 224],
        "padding": [left, top, 224 - w - left, 224 - h - top],
        "A": [[sx, 0, tx], [0, sy, ty], [0, 0, 1]],
        "coordinate_convention": convention,
        "source_coordinate_convention_verified": False,
        "method": "Pillow.ImageOps.contain/BILINEAR/center/black",
    }
    return np.asarray(canvas), geometry


def transform_points(points, geometry):
    a = np.asarray(geometry["A"], dtype=np.float64)
    # Axis-wise operations preserve an unknown axis without contaminating the other.
    return points * np.diag(a)[:2] + a[:2, 2]


def content_mask(geometry):
    mask = np.zeros((224, 224), dtype=bool)
    left, top = geometry["padding"][:2]
    w, h = geometry["content_size"]
    mask[top : top + h, left : left + w] = True
    return mask


def select_camera(full, frame_ids):
    """Exact sparse lookup; never interpolate or index by selection position."""
    ids = np.asarray(full["camera_pose_indices"])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate camera frame indices")
    lookup = {int(v): i for i, v in enumerate(ids)}
    if any(int(f) not in lookup for f in frame_ids):
        raise ValueError("camera poses unavailable for selected frames")
    poses = np.asarray(full["camera_poses"])[[lookup[int(f)] for f in frame_ids]].copy()
    static = full["camera_intrinsics_static"]
    if len(static) != 1 or len(full["camera_intrinsics_dynamic"]):
        raise ValueError("JPEG224 MolmoSpaces requires one source pixel-domain static K")
    k = np.asarray(static[0], dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("invalid source pixel-domain K")
    return poses, np.repeat(k[None], len(frame_ids), axis=0)
