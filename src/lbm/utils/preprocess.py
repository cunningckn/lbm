"""Shared state/action normalization and image preprocessing."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lbm.action_space import actions_in_train_space, column_scale_mask, uses_rel
from lbm.utils.quantiles import DiskQuantiles

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_norm_stats(raw):
    stats = raw["norm_stats"] if "norm_stats" in raw else raw
    if "state" not in stats and "actions" not in stats:
        key = "xdof" if "xdof" in stats else next(iter(stats))
        stats = stats[key]
    return {key: {k: np.asarray(v, dtype=np.float32) for k, v in stats[key].items()} for key in ("state", "actions")}


def load_norm_stats(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_norm_stats(raw)


def _format_hz_or_s(value: float) -> str:
    x = float(value)
    if x <= 0:
        raise ValueError(f"expected a positive value, got {value}")
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:g}"


def format_action_freq_hz(action_freq: float) -> str:
    return _format_hz_or_s(action_freq)


def _action_type_tag(slices: tuple) -> str:
    """``eef_delta_xyz_rotvec`` / ``joint_rel_default`` from the first non-gripper group."""
    from lbm.action_space import GRIPPER

    picked = next((sl for sl in slices if sl.kind != GRIPPER), None)
    if picked is None and slices:
        picked = slices[0]
    if picked is None:
        return ""
    fmt = getattr(picked, "format", "") or "default"
    return f"{picked.kind}_{picked.rep}_{fmt}"


def norm_stats_filename(
    action_freq: float,
    action_length: float | None = None,
    *,
    slices: tuple = (),
) -> str:
    hz = format_action_freq_hz(action_freq)
    tag = _action_type_tag(slices)
    prefix = f"norm_stats_{tag}_" if tag else "norm_stats_"
    if action_length is not None and uses_rel(slices):
        return f"{prefix}{hz}hz_{_format_hz_or_s(action_length)}s.json"
    return f"{prefix}{hz}hz.json"


def dump_norm_stats_path(root, action_freq: float, action_length: float, slices: tuple) -> Path:
    return Path(root) / norm_stats_filename(action_freq, action_length, slices=slices)


def load_dump_norm_stats(root, action_freq: float, action_length: float, slices: tuple) -> dict | None:
    """Matching ``norm_stats_{kind}_{rep}_{format}_{freq}hz.json`` (plus length for rel)."""
    if root is None:
        return None
    path = dump_norm_stats_path(root, action_freq, action_length, slices)
    if not path.is_file():
        return None
    return load_norm_stats(path)


def save_norm_stats(path, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as out:
            tmp = Path(out.name)
            out.write(json.dumps(_jsonable(stats), indent=2, allow_nan=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


class _Moments:
    def __init__(self, *, scratch_dir=None, block_rows=65536) -> None:
        self.quantiles = DiskQuantiles(directory=scratch_dir, block_rows=block_rows)
        self.count = 0
        self.sum = None
        self.sumsq = None
        self.min = None
        self.max = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.quantiles.close()

    def update(self, arr) -> None:
        x = np.asarray(arr)
        if x.size == 0:
            return
        if x.ndim == 1:
            x = x[None]
        x = x.reshape(-1, x.shape[-1])
        dim = int(x.shape[-1])
        if self.sum is not None and dim != len(self.sum):
            raise ValueError("normalization dimension changed between batches")
        # Validate before updating counts; all temporary arrays stay block-sized.
        for start in range(0, len(x), self.quantiles.block_rows):
            if not np.isfinite(x[start:start + self.quantiles.block_rows]).all():
                raise ValueError("normalization data must be finite")
        if self.sum is None:
            self.sum = np.zeros(dim, dtype=np.float64)
            self.sumsq = np.zeros(dim, dtype=np.float64)
            self.min = np.full(dim, np.inf)
            self.max = np.full(dim, -np.inf)
        for start in range(0, len(x), self.quantiles.block_rows):
            block = np.asarray(x[start:start + self.quantiles.block_rows], dtype=np.float64)
            self.quantiles.append(block)
            self.count += len(block)
            self.sum += block.sum(axis=0)
            self.sumsq += np.square(block).sum(axis=0)
            self.min = np.minimum(self.min, block.min(axis=0))
            self.max = np.maximum(self.max, block.max(axis=0))

    def as_dict(self) -> dict:
        if self.sum is None or self.count <= 0:
            raise ValueError("no state/action frames")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        q01, q99 = self.quantiles.quantiles([0.01, 0.99])
        return {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
            "count": int(self.count),
            "quantile_fallback": 1,
        }


def compute_norm_stats(dataset, *, progress: bool | None = None, leave: bool = True, scratch_dir=None) -> dict:
    """Mean/std, min/max, and q01/q99 over training-space state/action rows.

    ``rel`` slides every window (same gather as ``__getitem__``). ``abs`` / ``delta``
    use the packed episode (consecutive delta is an approximation of chunk stats).
    Flatten every chunk step as a row ``(N, D)``.
    """
    from lbm.dataloader.custom.fk_cache import require_fk_cache

    require_fk_cache(dataset)
    with _Moments(scratch_dir=scratch_dir) as state_m, _Moments(scratch_dir=scratch_dir) as action_m:
        n_episodes = len(dataset.episodes) if dataset.episodes else len(dataset.records)
        spec = dataset.spec
        root = dataset.root
        dump = Path(root).name if root is not None else spec.name
        indices = range(n_episodes)
        if progress is not False:
            from lbm.utils.progress import track

            indices = track(indices, desc=f"norm {dump}", total=n_episodes, unit="ep", leave=leave)
        slide = uses_rel(getattr(dataset, "_space", ()))
        for epi_i in indices:
            state, action, slices = dataset._policy_vectors(epi_i)
            n = int(action.shape[0])
            if n == 0:
                continue
            if slide:
                deltas = dataset.action_deltas
                for t in range(n):
                    chunk = dataset._gather_vector(action, t, deltas, n)
                    st = state[min(t, n - 1)]
                    chunk = actions_in_train_space(chunk, st, slices)
                    action_m.update(chunk)
                    state_m.update(st)
            else:
                packed = actions_in_train_space(action, state[0], slices)
                state_m.update(state)
                action_m.update(packed)
        payload = {
            "norm_stats": {
                "state": state_m.as_dict(),
                "actions": action_m.as_dict(),
            },
            "action_space": [sl.as_dict() for sl in getattr(dataset, "_space", ())],
            "action_mode": dataset.action_mode,
            "action_kind": getattr(dataset, "action_kind", None),
            "action_format": getattr(dataset, "action_format", None),
            "action_freq": float(dataset.action_freq),
            "action_length": float(dataset.action_length),
            "spec": spec.name,
            "embodiment": spec.embodiment,
            "n_episodes": int(n_episodes),
        }
        if root is not None:
            payload["dataset_root"] = str(root)
        return payload


def _has_quantiles(stats) -> bool:
    return "q01" in stats and "q99" in stats


def _broadcast_stat(x, value):
    if torch.is_tensor(x):
        t = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        if t.ndim == 1 and x.ndim > 1:
            t = t.reshape((1,) * (x.ndim - 1) + (-1,))
        return t
    return np.asarray(value, dtype=np.float32)


def _quantile_bounds(x, stats):
    low = _broadcast_stat(x, stats["q01"])
    high = _broadcast_stat(x, stats["q99"])
    # Opt-in metadata preserves the interpretation of existing checkpoint stats.
    if stats.get("quantile_fallback", 0) == 1:
        minimum = _broadcast_stat(x, stats["min"])
        maximum = _broadcast_stat(x, stats["max"])
        where = torch.where if torch.is_tensor(x) else np.where
        constant = maximum - minimum <= 1e-6
        fallback_low = where(constant, minimum - 0.5, minimum)
        fallback_high = where(constant, maximum + 0.5, maximum)
        degenerate = high - low <= 1e-6
        low = where(degenerate, fallback_low, low)
        high = where(degenerate, fallback_high, high)
    return low, high


def normalize(x, stats, slices=None, *, field: str = "action"):
    """Map ``x`` into model space. Prefers ``[q01, q99] → [-1, 1]``.

    ``slices`` skip EEF rot6d/quat rotation columns (already in ``[-1, 1]``).
    """
    if _has_quantiles(stats):
        q01, q99 = _quantile_bounds(x, stats)
        out = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    else:
        mean = _broadcast_stat(x, stats["mean"])
        std = _broadcast_stat(x, stats["std"])
        out = (x - mean) / (std + 1e-6)
    if not slices:
        return out
    dim = int(np.asarray(stats.get("q01", stats.get("mean"))).reshape(-1).shape[0])
    mask = column_scale_mask(slices, dim, field=field)
    return _apply_mask(x, out, mask)


def unnormalize(x, stats, slices=None, *, field: str = "action"):
    """Inverse of :func:`normalize`."""
    if _has_quantiles(stats):
        q01, q99 = _quantile_bounds(x, stats)
        out = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    else:
        mean = _broadcast_stat(x, stats["mean"])
        std = _broadcast_stat(x, stats["std"])
        out = x * (std + 1e-6) + mean
    if not slices:
        return out
    dim = int(np.asarray(stats.get("q01", stats.get("mean"))).reshape(-1).shape[0])
    mask = column_scale_mask(slices, dim, field=field)
    return _apply_mask(x, out, mask)


def _apply_mask(original, transformed, mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if torch.is_tensor(original):
        m = torch.as_tensor(mask, device=original.device)
        while m.ndim < original.ndim:
            m = m.reshape((1,) * (original.ndim - m.ndim) + m.shape)
        return torch.where(m, transformed, original)
    m = mask
    while m.ndim < original.ndim:
        m = np.expand_dims(m, axis=0)
    return np.where(m, transformed, original)


def resize_with_pad(img_hwc, target_h=224, target_w=224):
    h, w, _ = img_hwc.shape
    if (h, w) == (target_h, target_w):
        return img_hwc
    ratio = max(w / target_w, h / target_h)
    new_h = max(1, int(round(h / ratio)))
    new_w = max(1, int(round(w / ratio)))
    resized = F.interpolate(
        img_hwc.permute(2, 0, 1).unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    pad_h0 = (target_h - new_h) // 2
    pad_h1 = target_h - new_h - pad_h0
    pad_w0 = (target_w - new_w) // 2
    pad_w1 = target_w - new_w - pad_w0
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), value=0)
    return padded.permute(1, 2, 0)


def imagenet_normalize(img_chw):
    mean = IMAGENET_MEAN.to(device=img_chw.device, dtype=img_chw.dtype)
    std = IMAGENET_STD.to(device=img_chw.device, dtype=img_chw.dtype)
    return (img_chw - mean) / (std + 1e-6)


def resize_pad_normalize(img_chw, target_h=224, target_w=224):
    x = torch.as_tensor(img_chw).float()
    if x.max() > 1.0:
        x = x / 255.0
    x = resize_with_pad(x.permute(1, 2, 0), target_h, target_w).permute(2, 0, 1)
    return imagenet_normalize(x)
