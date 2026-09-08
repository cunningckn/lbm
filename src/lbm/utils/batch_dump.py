"""Dump a collated loader batch for visual inspection.

Vision → PNG + MP4, state/action → plots, language → txt. Used by
``scripts/inspect_data.py`` and optionally at training start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

__all__ = ["describe_loader_batch", "save_loader_batch", "video_fps_from_config"]


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def video_fps_from_config(config, dataset=None) -> float:
    """Pick a reasonable MP4 fps from temporal config or dump specs."""
    hist = float(getattr(getattr(config, "model", config), "history_freq", 0) or 0)
    if hist > 0:
        return hist
    datasets = getattr(dataset, "datasets", None)
    specs = []
    if datasets:
        specs = [getattr(ds, "spec", None) for ds in datasets]
    elif dataset is not None:
        specs = [getattr(dataset, "spec", None)]
    for spec in specs:
        fps = float(getattr(spec, "fps", 0) or 0)
        if fps > 0:
            return fps
    return 10.0


def describe_loader_batch(batch: dict[str, Any]) -> str:
    """Human-readable shapes / dtypes for a collated dump batch."""
    lines = []
    for key, value in batch.items():
        if torch.is_tensor(value):
            lines.append(f"{key}: {tuple(value.shape)} {value.dtype}")
        elif isinstance(value, (list, tuple)):
            preview = value[:4] if len(value) > 4 else value
            lines.append(f"{key}: {type(value).__name__}[{len(value)}] {preview!r}")
        else:
            lines.append(f"{key}: {value!r}")
    return "\n".join(lines)


def _write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    import cv2

    arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] < 2:
        arr = np.repeat(arr, 2, axis=0)
    _, h, w, _ = arr.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(max(fps, 1.0)), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {path}")
    try:
        for frame in arr:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _palette(n: int) -> list[tuple[int, int, int]]:
    import colorsys

    n = max(int(n), 1)
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i / n) % 1.0, 0.75, 0.88)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    xs: list[int],
    ys: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int],
) -> None:
    pts: list[tuple[int, int]] = []
    for i, x in enumerate(xs):
        if mask is not None and not bool(mask[i]):
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)
            pts = []
            continue
        pts.append((x, int(ys[i])))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=2)
    elif len(pts) == 1:
        x, y = pts[0]
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)


def _plot_series(
    path: Path,
    arr: np.ndarray,
    mask: np.ndarray | None,
    title: str,
    *,
    overlay: np.ndarray | None = None,
    overlay_mask: np.ndarray | None = None,
) -> None:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 1:
        x = x[None]
    if x.size == 0:
        return
    m = None if mask is None else np.asarray(mask, dtype=bool).reshape(x.shape)
    extra = None if overlay is None else np.asarray(overlay, dtype=np.float64)
    extra_m = None if overlay_mask is None else np.asarray(overlay_mask, dtype=bool)
    t_len, dim = x.shape
    width, height = 1100, 420
    left, right, top, bottom = 56, 16, 36, 40
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((left, 8), title, fill=(20, 20, 20))
    plot_w = width - left - right
    plot_h = height - top - bottom
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(180, 180, 180))

    chunks = [x[m] if m is not None else x.ravel()]
    if extra is not None:
        chunks.append(extra[extra_m] if extra_m is not None else extra.ravel())
    valid = np.concatenate([c.ravel() for c in chunks if np.asarray(c).size])
    if valid.size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return
    ymin, ymax = float(np.min(valid)), float(np.max(valid))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    pad = 0.05 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad

    def y_px(val: float) -> int:
        return int(top + plot_h * (1.0 - (val - ymin) / (ymax - ymin)))

    colors = _palette(dim)
    if t_len == 1 and extra is None:
        bar_w = max(1, plot_w // max(dim, 1) - 2)
        for d in range(dim):
            if m is not None and not bool(m[0, d]):
                continue
            x0 = left + int((d + 0.15) * plot_w / dim)
            y1 = y_px(float(x[0, d]))
            y0 = y_px(0.0) if ymin < 0 < ymax else top + plot_h
            draw.rectangle([x0, min(y0, y1), x0 + bar_w, max(y0, y1)], fill=colors[d])
        draw.text((left, height - 24), f"dim 0..{dim - 1}", fill=(80, 80, 80))
    else:
        t_axis = t_len if extra is None else int(extra.shape[1])
        xs = [left + int(i * (plot_w - 1) / max(t_axis - 1, 1)) for i in range(t_axis)]
        series = extra if extra is not None else x[None]
        series_m = extra_m if extra is not None else (None if m is None else m[None])
        for b in range(series.shape[0]):
            for d in range(dim):
                ys = np.array([y_px(float(v)) for v in series[b, :t_axis, d]])
                dm = None if series_m is None else series_m[b, :t_axis, d]
                _draw_lines(draw, xs, ys, dm, colors[d])
        draw.text((left, height - 24), f"t=0..{t_axis - 1}  dims={dim}", fill=(80, 80, 80))
    if dim <= 16:
        legend_x = width - 90
        for d, color in enumerate(colors):
            ly = top + 4 + d * 14
            if ly > top + plot_h - 8:
                break
            draw.rectangle([legend_x, ly, legend_x + 10, ly + 8], fill=color)
            draw.text((legend_x + 14, ly - 2), f"d{d}", fill=(40, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_loader_batch(batch: dict[str, Any], out_dir: str | Path, *, fps: float = 10.0) -> Path:
    """Write vision / state / action / language artifacts for one loader batch.

    Expects the collated dump format (``image``, ``action``, ``lang``, …).
    Returns the output directory.
    """
    if "image" not in batch or "action" not in batch:
        raise ValueError("save_loader_batch expects a collated dump batch with 'image' and 'action'")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = _as_numpy(batch["image"])
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # (B, C, T, H, W, 3)
    bsz, n_cams, t_hist = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    cams = list(batch.get("camera_keys") or [f"cam{i}" for i in range(n_cams)])
    cam_mask = _as_numpy(batch["camera_mask"]) if "camera_mask" in batch else np.ones((bsz, n_cams), dtype=bool)
    langs = list(batch.get("lang") or [""] * bsz)
    tags = list(batch.get("robot_tag") or [""] * bsz)

    action = _as_numpy(batch["action"])
    action_mask = _as_numpy(batch["action_mask"]) if "action_mask" in batch else None
    state = _as_numpy(batch["state"]) if "state" in batch else None
    state_mask = _as_numpy(batch["state_mask"]) if "state_mask" in batch else None

    lang_lines = []
    for i in range(bsz):
        tag = tags[i] if i < len(tags) else ""
        text = langs[i] if i < len(langs) else ""
        lang_lines.append(f"[{i}] tag={tag}  {text}")
    (out_dir / "lang.txt").write_text("\n".join(lang_lines) + "\n", encoding="utf-8")
    (out_dir / "meta.txt").write_text(describe_loader_batch(batch) + "\n", encoding="utf-8")

    img_dir = out_dir / "images"
    vid_dir = out_dir / "videos"
    plot_dir = out_dir / "plots"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    for b in range(bsz):
        sample_lang = langs[b] if b < len(langs) else ""
        (out_dir / f"lang_b{b:02d}.txt").write_text(str(sample_lang) + "\n", encoding="utf-8")
        for c in range(n_cams):
            if c < cam_mask.shape[1] and not bool(cam_mask[b, c]):
                continue
            cam = _safe(cams[c] if c < len(cams) else f"cam{c}")
            frames = image[b, c]
            for t in range(t_hist):
                Image.fromarray(frames[t]).save(img_dir / f"b{b:02d}_{cam}_t{t:03d}.png")
            _write_video(vid_dir / f"b{b:02d}_{cam}.mp4", frames, fps)
        _plot_series(
            plot_dir / f"action_b{b:02d}.png",
            action[b],
            None if action_mask is None else action_mask[b],
            f"action sample {b}  shape={tuple(action[b].shape)}",
        )
        if state is not None:
            _plot_series(
                plot_dir / f"state_b{b:02d}.png",
                state[b],
                None if state_mask is None else state_mask[b],
                f"state sample {b}  shape={tuple(state[b].shape)}",
            )

    _plot_series(
        plot_dir / "action_batch.png",
        action[0],
        None if action_mask is None else action_mask[0],
        f"action overlay  B={bsz} T={action.shape[1]} D={action.shape[2]}",
        overlay=action,
        overlay_mask=action_mask,
    )
    if state is not None:
        _plot_series(
            plot_dir / "state_batch.png",
            state[0],
            None if state_mask is None else state_mask[0],
            f"state overlay  B={bsz} T={state.shape[1]} D={state.shape[2]}",
            overlay=state,
            overlay_mask=state_mask,
        )

    cell_h, cell_w = int(image.shape[3]), int(image.shape[4])
    sheet = Image.new("RGB", (n_cams * cell_w, bsz * cell_h), (0, 0, 0))
    for b in range(bsz):
        for c in range(n_cams):
            if c < cam_mask.shape[1] and not bool(cam_mask[b, c]):
                continue
            sheet.paste(Image.fromarray(image[b, c, -1]), (c * cell_w, b * cell_h))
    sheet.save(out_dir / "overview.png")
    return out_dir
