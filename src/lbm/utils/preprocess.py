"""Shared state/action normalization and image preprocessing."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lbm.action_space import apply_action_space_frames, resolve_action_space

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
    """``eef_delta`` / ``joint_rel`` from the first non-gripper group."""
    from lbm.action_space import GRIPPER

    picked = next((sl for sl in slices if sl.kind != GRIPPER), None)
    if picked is None and slices:
        picked = slices[0]
    if picked is None:
        return ""
    return f"{picked.kind}_{picked.rep}"


def norm_stats_filename(
    action_freq: float,
    action_length: float | None = None,
    *,
    slices: tuple = (),
) -> str:
    from lbm.action_space import uses_computed_delta

    hz = format_action_freq_hz(action_freq)
    tag = _action_type_tag(slices)
    prefix = f"norm_stats_{tag}_" if tag else "norm_stats_"
    if action_length is not None and uses_computed_delta(slices):
        return f"{prefix}{hz}hz_{_format_hz_or_s(action_length)}s.json"
    return f"{prefix}{hz}hz.json"


def dump_norm_stats_path(root, action_freq: float, action_length: float, slices: tuple) -> Path:
    return Path(root) / norm_stats_filename(action_freq, action_length, slices=slices)


def load_dump_norm_stats(root, action_freq: float, action_length: float, slices: tuple) -> dict | None:
    """Matching ``norm_stats_{kind}_{rep}_{freq}hz.json`` (plus length for computed delta)."""
    if root is None:
        return None
    path = dump_norm_stats_path(root, action_freq, action_length, slices)
    if not path.is_file():
        return None
    return load_norm_stats(path)


def save_norm_stats(path, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(stats), indent=2) + "\n", encoding="utf-8")


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
    def __init__(self) -> None:
        self.count = 0
        self.sum = None
        self.sumsq = None
        self.min = None
        self.max = None
        self._chunks: list[np.ndarray] = []

    def update(self, arr) -> None:
        x = np.asarray(arr, dtype=np.float64)
        if x.size == 0:
            return
        if x.ndim == 1:
            x = x[None]
        x = x.reshape(-1, x.shape[-1])
        dim = int(x.shape[-1])
        if self.sum is None:
            self.sum = np.zeros(dim, dtype=np.float64)
            self.sumsq = np.zeros(dim, dtype=np.float64)
            self.min = np.full(dim, np.inf)
            self.max = np.full(dim, -np.inf)
        self.count += int(x.shape[0])
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)
        self.min = np.minimum(self.min, x.min(axis=0))
        self.max = np.maximum(self.max, x.max(axis=0))
        self._chunks.append(x.astype(np.float32, copy=False))

    def as_dict(self) -> dict:
        if self.sum is None or self.count <= 0:
            raise ValueError("no state/action frames")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        stacked = np.concatenate(self._chunks, axis=0)
        q01, q99 = np.quantile(stacked, [0.01, 0.99], axis=0)
        return {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
            "count": int(self.count),
        }


def compute_norm_stats(dataset, *, progress: bool | None = None, leave: bool = True) -> dict:
    """Mean/std, min/max, and q01/q99 over every frame of ``state`` and ``action``.

    Output is readable by :func:`parse_norm_stats` / :func:`load_norm_stats`.
    ``normalize`` maps ``[q01, q99]`` to ``[-1, 1]`` when those keys exist.
    """
    state_m = _Moments()
    action_m = _Moments()
    n_episodes = len(dataset.episodes) if dataset.episodes else len(dataset.records)
    spec = dataset.spec
    root = dataset.root
    dump = Path(root).name if root is not None else spec.name
    indices = range(n_episodes)
    if progress is not False:
        from lbm.utils.progress import track

        indices = track(indices, desc=f"norm {dump}", total=n_episodes, unit="ep", leave=leave)
    slices = resolve_action_space(spec, dataset.action_mode)
    hop = int(dataset.action_hop)
    for epi_i in indices:
        state, action = dataset._vectors(epi_i)
        action = apply_action_space_frames(action, state, slices, hop=hop)
        state_m.update(state)
        action_m.update(action)
    payload = {
        "norm_stats": {
            "state": state_m.as_dict(),
            "actions": action_m.as_dict(),
        },
        "action_space": [sl.as_dict() for sl in slices],
        "action_mode": dataset.action_mode,
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


def normalize(x, stats):
    """Map ``x`` into model space. Prefers ``[q01, q99] → [-1, 1]``."""
    if _has_quantiles(stats):
        q01 = _broadcast_stat(x, stats["q01"])
        q99 = _broadcast_stat(x, stats["q99"])
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    mean = _broadcast_stat(x, stats["mean"])
    std = _broadcast_stat(x, stats["std"])
    return (x - mean) / (std + 1e-6)


def unnormalize(x, stats):
    """Inverse of :func:`normalize`."""
    if _has_quantiles(stats):
        q01 = _broadcast_stat(x, stats["q01"])
        q99 = _broadcast_stat(x, stats["q99"])
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    mean = _broadcast_stat(x, stats["mean"])
    std = _broadcast_stat(x, stats["std"])
    return x * (std + 1e-6) + mean


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
