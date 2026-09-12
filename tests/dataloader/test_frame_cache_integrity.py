"""Corruption must be rejected before a frame cache is marked ready."""

import json

import numpy as np
import pytest

from lbm.dataloader.mmap import frame_mmap_io as cache
from lbm.dataloader.mmap.mmap_io import read_manifest


@pytest.mark.parametrize("damage", ["truncate", "offset", "length", "count", "dtype", "utf8"])
def test_corrupt_pack_rejected_and_rebuilt(tmp_path, damage):
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    cache.write_frame_cache(tmp_path, source_tag="source", frames=frames, progress=False)
    if damage == "truncate":
        p = tmp_path / "frames.bin"
        p.write_bytes(p.read_bytes()[:-1])
    elif damage == "offset":
        arr = np.load(tmp_path / "offset.npy")
        arr[1] += 1
        np.save(tmp_path / "offset.npy", arr)
    elif damage == "length":
        np.save(tmp_path / "length.npy", np.zeros(3, dtype=np.uint32))
    elif damage == "dtype":
        np.save(tmp_path / "offset.npy", np.zeros(3, dtype=np.float64))
    elif damage == "count":
        p = tmp_path / cache.MANIFEST
        data = json.loads(p.read_text())
        data["num_frames"] = 4
        p.write_text(json.dumps(data))
    else:
        (tmp_path / cache.MANIFEST).write_bytes(b"\xff")
        assert read_manifest(tmp_path / cache.MANIFEST) is None
    assert not cache._cache_is_ready(tmp_path, "source")
    with pytest.raises((ValueError, FileNotFoundError)):
        cache.MmapFrameEpisode.open(tmp_path)
    cache.write_frame_cache(tmp_path, source_tag="source", frames=frames, progress=False)
    assert cache._cache_is_ready(tmp_path, "source")
    episode = cache.MmapFrameEpisode.open(tmp_path)
    try:
        assert episode.get_frames(np.array([0, 2])).shape == (2, 8, 8, 3)
    finally:
        episode.close()


def test_failed_disk_commit_leaves_no_ready_cache(tmp_path, monkeypatch):
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    cache.write_frame_cache(tmp_path, source_tag="source", frames=frames, progress=False)

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(cache.os, "fsync", disk_full)
        with pytest.raises(OSError, match="No space"):
            cache.write_frame_cache(tmp_path, source_tag="source", frames=frames, progress=False)
    assert not cache._cache_is_ready(tmp_path, "source")
    assert not (tmp_path / "frames.bin.tmp").exists()
    cache.write_frame_cache(tmp_path, source_tag="source", frames=frames, progress=False)
    assert cache._cache_is_ready(tmp_path, "source")


def test_killed_writer_is_recoverable(tmp_path):
    import select
    import subprocess
    import sys

    script = """
import sys, time
from pathlib import Path
from lbm.dataloader.mmap.frame_mmap_io import _commit_jpeg_pack

def frames():
    yield b'x' * 8192
    print('writing', flush=True)
    time.sleep(300)
    yield b'y' * 8192

_commit_jpeg_pack(Path(sys.argv[1]), frames(), source_tag='source',
                  jpeg_quality=85, image_size=None, height=8, width=8)
"""
    proc = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        ready, _, _ = select.select([proc.stdout], [], [], 60)
        assert ready, "cache writer did not start"
        assert proc.stdout.readline().strip() == "writing"
        proc.kill()
        proc.wait(timeout=10)
        assert not cache._cache_is_ready(tmp_path, "source")
        assert (tmp_path / "frames.bin.tmp").exists()
        cache.write_frame_cache(tmp_path, source_tag="source",
                                frames=np.zeros((2, 8, 8, 3), dtype=np.uint8), progress=False)
        assert cache._cache_is_ready(tmp_path, "source")
        assert not (tmp_path / "frames.bin.tmp").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        proc.stdout.close()
