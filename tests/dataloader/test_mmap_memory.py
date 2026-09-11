"""Bounded memory and failure recovery for JPEG cache construction."""

from concurrent.futures import Future
import json
import tracemalloc

import numpy as np
import pytest

from lbm.dataloader.mmap import frame_mmap_io as cache


def test_encoded_payload_is_not_retained(tmp_path, monkeypatch):
    # 64 MiB of independent encoded frames; only one should be resident at a time.
    monkeypatch.setattr(cache, "_encode_rgb_jpeg", lambda *a, **k: (b"x" * (256 * 1024), (8, 8)))
    monkeypatch.setattr(cache, "reclaim_if_over", lambda: None)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    tracemalloc.start()
    try:
        cache.write_frame_cache(tmp_path, source_tag="memory", frame_iter=(frame for _ in range(256)), progress=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024
    assert (tmp_path / "frames.bin").stat().st_size == 64 * 1024 * 1024
    np.testing.assert_array_equal(np.load(tmp_path / "offset.npy"), np.arange(256) * 256 * 1024)


def test_failed_stream_does_not_publish_and_retry_succeeds(tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def broken():
        yield frame
        yield frame
        raise RuntimeError("decoder interrupted")

    with pytest.raises(RuntimeError, match="decoder interrupted"):
        cache.write_frame_cache(tmp_path, source_tag="retry", frame_iter=broken(), progress=False)
    assert not (tmp_path / cache.MANIFEST).exists()
    assert not (tmp_path / "frames.bin.tmp").exists()
    # Simulate a leftover temporary payload after SIGKILL.
    (tmp_path / "frames.bin.tmp").write_bytes(b"stale")
    cache.write_frame_cache(tmp_path, source_tag="retry", frame_iter=iter([frame]), progress=False)
    manifest = json.loads((tmp_path / cache.MANIFEST).read_text())
    assert manifest["num_frames"] == 1
    episode = cache.MmapFrameEpisode.open(tmp_path)
    try:
        assert len(episode.gather_blobs(np.array([0]))) == 1
    finally:
        episode.close()


def test_bounded_submit_does_not_materialize_all_tasks():
    submitted = consumed = 0

    class CountedFuture(Future):
        def result(self, timeout=None):
            nonlocal consumed
            consumed += 1
            return super().result(timeout)

    class Pool:
        def submit(self, fn, payload):
            nonlocal submitted
            submitted += 1
            assert submitted - consumed <= 4
            future = CountedFuture()
            future.set_result(fn(payload))
            return future

    results = list(cache._run_bounded(Pool(), ((int, i) for i in range(1000)), limit=4))
    assert sorted(results) == list(range(1000))


def test_shared_spool_keeps_payload_on_disk(tmp_path):
    spool = cache._JpegSpool()
    tracemalloc.start()
    try:
        for _ in range(128):
            spool.append(b"x" * (256 * 1024), directory=tmp_path)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 8 * 1024 * 1024
        assert len(spool) == 128
        assert sum(len(blob) for blob in spool) == 32 * 1024 * 1024
    finally:
        spool.clear()
        tracemalloc.stop()
    assert spool.file is None
