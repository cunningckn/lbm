from types import SimpleNamespace

import numpy as np
import pytest

from lbm.dataloader.custom import video


@pytest.fixture
def gop_video(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "gop.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 30
        stream.codec_context.max_b_frames = 0
        for i in range(30):
            pixels = np.full((32, 32, 3), 20 + i * 4, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def test_random_seek_decodes_target_not_preceding_keyframe(gop_video):
    indices = [29, 15, 17, 15, 0, -2, 99]
    expected = [136, 80, 88, 80, 20, 20, 136]
    for read in (video._read_av, video._read_cv2):
        frames = read(gop_video, indices, (32, 32))
        assert frames is not None
        np.testing.assert_allclose(frames.mean(axis=(1, 2, 3)), expected, atol=3)
    frames = video.read_mp4_span(gop_video, 15, 18)
    np.testing.assert_allclose(frames.mean(axis=(1, 2, 3)), [80, 84, 88], atol=3)


@pytest.mark.parametrize("span", [False, True])
def test_unexpected_seek_failure_propagates_and_closes(monkeypatch, span):
    pytest.importorskip("av")
    closed = []

    def fail(*args, **kwargs):
        raise MemoryError("decoder allocation")

    container = SimpleNamespace(seek=fail, close=lambda: closed.append(True))
    stream = SimpleNamespace(average_rate=30, time_base=1 / 30, frames=30)
    monkeypatch.setattr(video, "_open_av", lambda path: (container, stream))
    with pytest.raises(MemoryError, match="decoder allocation"):
        if span:
            list(video._iter_av_span("unused", 15, 18))
        else:
            video._read_av("unused", [15], (32, 32))
    assert closed == [True]


def test_av_open_closes_container_without_video(monkeypatch):
    av = pytest.importorskip("av")
    closed = []
    container = SimpleNamespace(streams=SimpleNamespace(video=[]), close=lambda: closed.append(True))
    monkeypatch.setattr(av, "open", lambda path: container)
    assert video._open_av("unused") is None
    assert closed == [True]


def test_cv_decode_failure_releases_capture(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    closed = []

    def fail(*args):
        raise MemoryError("decoder allocation")

    cap = SimpleNamespace(isOpened=lambda: True, get=lambda prop: 30, set=fail,
                          release=lambda: closed.append(True))
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: cap)
    with pytest.raises(MemoryError):
        video._read_cv2("unused", [15], (32, 32))
    assert closed == [True]
