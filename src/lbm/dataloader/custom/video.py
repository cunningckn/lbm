"""Decode selected MP4 frames for custom dumps."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_mp4_indices(path: Path | str, indices: list[int], *, fallback_hw: tuple[int, int] = (224, 224)) -> np.ndarray:
    path = Path(path)
    h, w = fallback_hw
    if not path.is_file() or not indices:
        return np.zeros((len(indices), h, w, 3), dtype=np.uint8)
    frames = _read_av(path, indices, (h, w))
    if frames is not None and int(frames.max()) > 0:
        return frames
    cv = _read_cv2(path, indices, (h, w))
    if cv is not None and int(cv.max()) > 0:
        return cv
    if frames is not None:
        return frames
    if cv is not None:
        return cv
    return np.zeros((len(indices), h, w, 3), dtype=np.uint8)


def iter_mp4_all(video_path: str | Path, *, progress: bool = False):
    """Yield every native RGB frame. Av first, then OpenCV. Does not stack the episode."""
    path = Path(video_path)
    if not path.is_file():
        return
    yield from _prefer_av(_iter_av_all(path, progress=progress), _iter_cv2_all(path, progress=progress))


def read_mp4_all(video_path: str | Path, *args, progress: bool = False, **kwargs) -> np.ndarray:
    """Decode every frame for mmap cache build."""
    del args, kwargs
    frames = list(iter_mp4_all(video_path, progress=progress))
    if frames:
        return np.stack(frames, axis=0)
    return np.zeros((0, 1, 1, 3), dtype=np.uint8)


def contiguous_span(indices: list[int]) -> tuple[int, int] | None:
    """``[start, stop)`` if ``indices`` is ``start..stop-1``, else None."""
    if not indices:
        return None
    start = int(indices[0])
    if start < 0:
        return None
    for i, raw in enumerate(indices):
        if int(raw) != start + i:
            return None
    return start, start + len(indices)


def read_mp4_span(path: Path | str, start: int, stop: int, *, progress: bool = False) -> np.ndarray | None:
    """Sequential decode of ``[start, stop)``. None if the file cannot be opened."""
    path = Path(path)
    start, stop = int(start), int(stop)
    if not path.is_file() or stop <= start:
        return None
    frames = _read_av_span(path, start, stop, progress=progress)
    if frames is not None and frames.shape[0] == stop - start:
        return frames
    cv = _read_cv2_span(path, start, stop, progress=progress)
    if cv is not None and cv.shape[0] == stop - start:
        return cv
    return frames if frames is not None and frames.shape[0] else None


def iter_mp4_span(path: Path | str, start: int, stop: int, *, progress: bool = False):
    """Yield native RGB frames for file indices ``[start, stop)``. Av first, then OpenCV."""
    path = Path(path)
    start, stop = int(start), int(stop)
    if not path.is_file() or stop <= start:
        return
    yield from _prefer_av(
        _iter_av_span(path, start, stop, progress=progress),
        _iter_cv2_span(path, start, stop, progress=progress),
    )


def _prefer_av(av_frames, cv_frames):
    n = 0
    for frame in av_frames:
        n += 1
        yield frame
    if n:
        return
    yield from cv_frames


def _read_cv2(path: Path, indices: list[int], hw: tuple[int, int]) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out: list[np.ndarray] = []
    last = None
    h, w = hw
    for raw in indices:
        idx = int(np.clip(raw, 0, max(n - 1, 0))) if n else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            if last is None:
                last = np.zeros((h, w, 3), dtype=np.uint8)
            out.append(last)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last = rgb
        out.append(rgb)
    cap.release()
    return np.stack(out, axis=0) if out else None


def _open_av(path: Path):
    try:
        import av
    except ImportError:
        return None
    try:
        av.logging.set_level(av.logging.ERROR)
    except Exception:
        pass
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
    except Exception:
        return None
    return container, stream


def _read_av(path: Path, indices: list[int], hw: tuple[int, int]) -> np.ndarray | None:
    opened = _open_av(path)
    if opened is None:
        return None
    container, stream = opened
    fps = float(stream.average_rate or 30.0)
    tb = float(stream.time_base) if stream.time_base else (1.0 / max(fps, 1.0))
    n = int(stream.frames or 0)
    h, w = hw
    unique: list[int] = []
    seen: set[int] = set()
    for raw in indices:
        idx = int(np.clip(raw, 0, max(n - 1, 0))) if n else int(max(raw, 0))
        if idx not in seen:
            unique.append(idx)
            seen.add(idx)
    decoded: dict[int, np.ndarray] = {}
    max_i = max(unique) if unique else 0
    if max_i <= 8:
        for i, frame in enumerate(container.decode(stream)):
            if i in seen:
                decoded[i] = frame.to_ndarray(format="rgb24")
            if i >= max_i:
                break
    else:
        for idx in unique:
            pts = int(idx / max(fps, 1e-6) / max(tb, 1e-12))
            try:
                container.seek(pts, stream=stream)
            except Exception:
                pass
            for frame in container.decode(stream):
                decoded[idx] = frame.to_ndarray(format="rgb24")
                break
    container.close()
    last = np.zeros((h, w, 3), dtype=np.uint8)
    out = []
    for raw in indices:
        idx = int(np.clip(raw, 0, max(n - 1, 0))) if n else int(max(raw, 0))
        if idx in decoded:
            last = decoded[idx]
        out.append(last)
    return np.stack(out, axis=0) if out else None


def _av_frame_index(frame, fps: float) -> int | None:
    if frame.pts is None:
        return None
    tb = float(frame.time_base) if frame.time_base else (1.0 / max(fps, 1.0))
    return int(round(float(frame.pts) * tb * fps))


def _read_av_span(path: Path, start: int, stop: int, *, progress: bool = False) -> np.ndarray | None:
    frames = list(_iter_av_span(path, start, stop, progress=progress))
    return np.stack(frames, axis=0) if frames else None


def _iter_av_span(path: Path, start: int, stop: int, *, progress: bool = False):
    """Seek once, then decode forward. Drops frames whose pts index is before ``start``."""
    opened = _open_av(path)
    if opened is None:
        return
    container, stream = opened
    fps = float(stream.average_rate or 30.0)
    tb = float(stream.time_base) if stream.time_base else (1.0 / max(fps, 1.0))
    if start > 0:
        try:
            pts = int(start / max(fps, 1e-6) / max(tb, 1e-12))
            container.seek(pts, stream=stream, backward=True, any_frame=False)
        except Exception:
            pass
    want = stop - start
    iterator = container.decode(stream)
    if progress:
        from lbm.utils.progress import track

        iterator = track(iterator, desc=f"decode {path.name}", total=want, unit="f", leave=False)
    n = 0
    try:
        for frame in iterator:
            idx = _av_frame_index(frame, fps)
            if idx is None:
                if start > 0 and n == 0:
                    continue
            elif idx < start:
                continue
            yield frame.to_ndarray(format="rgb24")
            n += 1
            if n >= want:
                break
    finally:
        container.close()


def _read_cv2_span(path: Path, start: int, stop: int, *, progress: bool = False) -> np.ndarray | None:
    frames = list(_iter_cv2_span(path, start, stop, progress=progress))
    return np.stack(frames, axis=0) if frames else None


def _iter_cv2_span(path: Path, start: int, stop: int, *, progress: bool = False):
    try:
        import cv2
    except ImportError:
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    want = stop - start
    bar = None
    if progress:
        from lbm.utils.progress import progress_bar

        bar = progress_bar(total=want, desc=f"decode {path.name}", unit="f", leave=False)
    try:
        for _ in range(want):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
        cap.release()


def _iter_av_all(path: Path, *, progress: bool = False):
    opened = _open_av(path)
    if opened is None:
        return
    container, stream = opened
    iterator = container.decode(stream)
    n = int(stream.frames or 0) or None
    if progress:
        from lbm.utils.progress import track

        iterator = track(iterator, desc=f"decode {path.name}", total=n, unit="f", leave=False)
    try:
        for frame in iterator:
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def _iter_cv2_all(path: Path, *, progress: bool = False):
    try:
        import cv2
    except ImportError:
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    bar = None
    if progress:
        from lbm.utils.progress import progress_bar

        bar = progress_bar(total=n, desc=f"decode {path.name}", unit="f", leave=False)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
        cap.release()
