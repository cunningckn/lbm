"""ABC-130K MCAP (H.264 / JPEG camera blobs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lbm.action_space import bimanual_joint
from lbm.dataloader.custom.common.arrays import fit_dim
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import WRISTS, CustomSpec, make_spec

NAME = "abc"
SPEC = make_spec("abc", "abc", WRISTS, 14, 14, 10.0, 20, kind="mcap", action_space=bimanual_joint())

_STATE = (
    ("/left-arm-state", 6),
    ("/left-ee-state", 1),
    ("/right-arm-state", 6),
    ("/right-ee-state", 1),
)
_ACTION = (
    ("/left-arm-action", 6),
    ("/left-ee-action", 1),
    ("/right-arm-action", 6),
    ("/right-ee-action", 1),
)
_CAM = {
    "cam_high": ("/top-left-camera", "/top-right-camera", "/top-camera"),
    "cam_left_wrist": ("/left-wrist-camera",),
    "cam_right_wrist": ("/right-wrist-camera",),
}
_SCALAR_TOPICS = {t for t, _ in _STATE + _ACTION}


def _resample_ticks(t0: int, t1: int, fps: float) -> np.ndarray:
    """Same grid ``read_vectors`` uses: ``arange(t0, t1+1, 1e9/fps)``."""
    tick = int(round(1e9 / float(fps)))
    if tick <= 0 or int(t1) < int(t0):
        return np.zeros(0, dtype=np.int64)
    return np.arange(int(t0), int(t1) + 1, tick, dtype=np.int64)


def _resampled_n_frames(t0: int, t1: int, fps: float) -> int:
    return int(_resample_ticks(t0, t1, fps).shape[0])


def _index_span(records: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not records:
        return None
    lo = hi = int(records[0][0])
    for ts, _off in records[1:]:
        t = int(ts)
        if t < lo:
            lo = t
        elif t > hi:
            hi = t
    return lo, hi


def _read_chunk_index_spans(handle, chunk, want: set[int]) -> dict[int, tuple[int, int]]:
    """One sequential read of a chunk's MessageIndex block (not per-channel seeks)."""
    from io import BytesIO

    from mcap.data_stream import ReadDataStream
    from mcap.opcode import Opcode
    from mcap.records import MessageIndex

    if not chunk.message_index_offsets:
        return {}
    # Block starts at the first index of *any* channel (cameras often precede scalars).
    start = min(int(off) for off in chunk.message_index_offsets.values())
    size = int(chunk.message_index_length)
    if size <= 0:
        return {}
    handle.seek(start)
    blob = handle.read(size)
    stream = ReadDataStream(BytesIO(blob))
    out: dict[int, tuple[int, int]] = {}
    while stream.count + 9 <= len(blob):
        opcode = stream.read1()
        length = stream.read8()
        begin = stream.count
        if begin + length > len(blob):
            break
        if opcode == Opcode.MESSAGE_INDEX:
            record = MessageIndex.read(stream)
            cid = int(record.channel_id)
            if cid in want:
                span = _index_span(record.records)
                if span is not None:
                    out[cid] = span
        consumed = stream.count - begin
        pad = length - consumed
        if pad > 0:
            stream.read(pad)
    return out


def _endpoint_chunks(chunk_indexes, want: set[int]):
    """Chunks that hold each channel's earliest and latest messages (time-ordered writer)."""
    first: dict[int, object] = {}
    last_start: dict[int, object] = {}
    last_end: dict[int, object] = {}
    for chunk in sorted(chunk_indexes, key=lambda item: int(item.message_start_time)):
        for raw_cid in chunk.message_index_offsets:
            cid = int(raw_cid)
            if cid not in want:
                continue
            if cid not in first:
                first[cid] = chunk
            last_start[cid] = chunk
            end = int(chunk.message_end_time)
            prev = last_end.get(cid)
            if prev is None or end >= int(prev.message_end_time):
                last_end[cid] = chunk
    return {id(chunk): chunk for chunk in (*first.values(), *last_start.values(), *last_end.values())}


def _scalar_overlap_ns(path: Path | str) -> tuple[int, int]:
    """First/last log_time of every state/action topic, from message indexes (not camera span).

    Reads MessageIndex only on the first/last chunks that contain each scalar channel
    (plus the max-end-time chunk if it differs). Same overlap as ``read_vectors``.
    """
    from mcap.reader import make_reader

    path = Path(path)
    with path.open("rb") as handle:
        summary = make_reader(handle).get_summary()
        if summary is None or not summary.chunk_indexes:
            raise ValueError(f"mcap has no chunk indexes: {path}")
        by_topic = {ch.topic: int(ch.id) for ch in summary.channels.values()}
        missing = sorted(_SCALAR_TOPICS - set(by_topic))
        if missing:
            raise ValueError(f"mcap missing topics {missing}: {path}")
        want = {by_topic[topic] for topic in _SCALAR_TOPICS}
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for chunk in _endpoint_chunks(summary.chunk_indexes, want).values():
            for cid, span in _read_chunk_index_spans(handle, chunk, want).items():
                lo, hi = span
                first[cid] = lo if cid not in first else min(first[cid], lo)
                last[cid] = hi if cid not in last else max(last[cid], hi)
        if want - set(first) or want - set(last):
            raise ValueError(f"mcap missing scalar message indexes: {path}")
        t0 = max(first[cid] for cid in want)
        t1 = min(last[cid] for cid in want)
        if t1 < t0:
            raise ValueError(f"mcap scalar streams do not overlap: {path}")
        return t0, t1


def _n_frames_mcap(path: Path | str, fps: float) -> int:
    try:
        t0, t1 = _scalar_overlap_ns(path)
    except OSError:
        return 0
    return _resampled_n_frames(t0, t1, fps)


def _lang_from_path(path: Path) -> str:
    return path.parent.parent.name.replace("_", " ")


def scan(root: Path, spec: CustomSpec, *, max_episodes: int | None = None) -> list[EpisodeRecord]:
    import os

    from lbm.dataloader.custom.common.fs import list_files, map_threads

    root = Path(root)
    walk_root = root / "data" if (root / "data").is_dir() else root
    items = list_files(
        walk_root,
        name="episode.mcap",
        dir_depth=3,
        max_files=max_episodes,
        desc=f"scan {spec.name}",
    )
    fps = float(spec.fps)
    workers = min(64, max(32, os.cpu_count() or 8))
    rows = map_threads(
        lambda path: (path, _n_frames_mcap(path, fps)),
        items,
        workers=workers,
        chunksize=32,
        desc=f"scan {spec.name} n_frames",
    )
    return [
        EpisodeRecord(kind="mcap", path=str(path), n_frames=n, lang=_lang_from_path(path))
        for path, n in rows
        if n > 1
    ]


def read_vectors(record: EpisodeRecord, spec: CustomSpec, **_kwargs) -> tuple[np.ndarray, np.ndarray]:
    scalars, _cams, _instr = _read_mcap_file(record.path, spec.fps, cameras=False)
    parts = []
    for topic, dim in _STATE + _ACTION:
        if topic not in scalars:
            raise KeyError(f"mcap missing {topic} in {record.path}")
        ts, vals = scalars[topic]
        parts.append((ts, fit_dim(vals, dim)))
    t0 = max(int(p[0][0]) for p in parts)
    t1 = min(int(p[0][-1]) for p in parts)
    ticks = _resample_ticks(t0, t1, spec.fps)
    cols = []
    for ts, vals in parts:
        idx = np.searchsorted(ts, ticks, side="right") - 1
        idx = np.clip(idx, 0, len(vals) - 1)
        cols.append(vals[idx])
    packed = np.concatenate(cols, axis=1)
    state = fit_dim(packed[:, : spec.state_dim], spec.state_dim)
    action = fit_dim(packed[:, spec.state_dim : spec.state_dim + spec.action_dim], spec.action_dim)
    return state, action


def read_frames(record: EpisodeRecord, spec: CustomSpec, cam: str, indices: list[int]) -> np.ndarray:
    _scalars, cams, _instr = _read_mcap_file(record.path, spec.fps)
    hw = (spec.image_size, spec.image_size)
    topics = _CAM.get(cam, ())
    stream = None
    for topic in topics:
        if topic in cams:
            stream = cams[topic]
            break
    if not stream:
        return np.zeros((len(indices),) + hw + (3,), dtype=np.uint8)
    ts = np.asarray([t for t, _b, _f in stream], dtype=np.int64)
    t0 = int(ts[0])
    tick = int(round(1e9 / spec.fps))
    packet_idx = []
    for i in indices:
        target = t0 + int(i) * tick
        packet_idx.append(int(np.clip(np.searchsorted(ts, target) - 1, 0, len(stream) - 1)))
    packets = [blob for _t, blob, _f in stream]
    fmt = str(stream[0][2] or "h264")
    decoded = _decode_mcap_packets(packets, packet_idx, fmt, hw)
    return np.stack([decoded.get(j, np.zeros(hw + (3,), dtype=np.uint8)) for j in packet_idx], axis=0)


def _decode_mcap_packets(
    packets: list[bytes], indices: list[int], fmt: str, hw: tuple[int, int]
) -> dict[int, np.ndarray]:
    need = sorted({int(i) for i in indices if 0 <= int(i) < len(packets)})
    out: dict[int, np.ndarray] = {}
    jpeg_ok = True
    for i in need:
        img = _decode_video_blob(packets[i], fmt)
        if img is None:
            jpeg_ok = False
            break
        out[i] = img
    if jpeg_ok and len(out) == len(need):
        return out
    out.update(_decode_annexb(packets, need, fmt))
    return out


def _decode_annexb(packets: list[bytes], need: list[int], fmt: str) -> dict[int, np.ndarray]:
    if not need:
        return {}
    try:
        import av
    except ImportError:
        return {}
    try:
        av.logging.set_level(av.logging.ERROR)
    except Exception:
        pass
    name = "hevc" if any(token in fmt.lower() for token in ("265", "hevc")) else "h264"
    try:
        codec = av.codec.CodecContext.create(name, "r")
    except Exception:
        return {}
    want = set(need)
    max_i = max(want)
    decoded: dict[int, np.ndarray] = {}
    for i, pkt in enumerate(packets):
        if i > max_i:
            break
        if not pkt:
            continue
        try:
            frames = list(codec.decode(av.Packet(pkt)))
        except Exception:
            continue
        if not frames or i not in want:
            continue
        frame = frames[0]
        try:
            rgb = frame.to_ndarray(channel_last=True, format="rgb24")
        except TypeError:
            rgb = frame.to_ndarray(format="rgb24")
        decoded[i] = rgb
        want.discard(i)
        if not want:
            break
    return decoded


def _decode_video_blob(blob: bytes, fmt: str) -> np.ndarray | None:
    import cv2

    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def _read_mcap_file(path: str, fps: float, *, cameras: bool = True):
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise ImportError("abc mcap reader needs mcap and mcap-protobuf-support") from exc

    cam_names = {t for ts in _CAM.values() for t in ts}
    scalar_names = {t for t, _ in _STATE + _ACTION}
    want = set(scalar_names)
    want.add("/instruction")
    if cameras:
        want |= cam_names
    cams: dict[str, list] = {}
    scalars: dict[str, list] = {}
    instruction = ""
    with open(path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages(topics=want):
            topic = channel.topic
            if topic == "/instruction" and not instruction:
                instruction = str(getattr(decoded, "data", "") or "")
            elif cameras and topic in cam_names:
                fmt = str(getattr(decoded, "format", "") or "h264").lower()
                cams.setdefault(topic, []).append((int(message.log_time), bytes(decoded.data), fmt))
            elif topic in scalar_names:
                pos = np.asarray(list(getattr(decoded, "position", []) or []), dtype=np.float32)
                scalars.setdefault(topic, []).append((int(message.log_time), pos))
    out_s = {}
    for topic, msgs in scalars.items():
        msgs.sort(key=lambda x: x[0])
        ts = np.asarray([t for t, _ in msgs], dtype=np.int64)
        vals = np.stack([v if v.ndim else np.asarray([v]) for _, v in msgs], axis=0)
        if vals.ndim == 1:
            vals = vals[:, None]
        out_s[topic] = (ts, vals)
    for msgs in cams.values():
        msgs.sort(key=lambda x: x[0])
    return out_s, cams, instruction
