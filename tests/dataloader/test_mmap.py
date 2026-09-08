"""Unit tests for mmap parquet + JPEG frame cache."""

from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lbm.dataloader.mmap.frame_mmap_io import (
    MmapFrameEpisode,
    MmapFrameStore,
    align_frames_to_parquet_steps,
    lerobot_v3_file_span,
    pack_jpeg_frames,
    partition_shared_frame_jobs,
    split_pending_frame_jobs,
    write_frame_cache,
    write_shared_mp4_caches,
)
from lbm.dataloader.mmap.jpeg_io import (
    decode_jpeg_rgb,
    decode_jpegs_into,
    encode_jpeg_rgb,
    resize_with_pad,
)
from lbm.dataloader.mmap.mmap_io import MmapTrajectoryStore, TrajectoryDataView


def _write_episode_parquet(path: Path, length: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "episode_index": np.zeros(length, dtype=np.int64),
            "timestamp": np.arange(length, dtype=np.float64) * 0.1,
            "observation.state": [np.arange(3, dtype=np.float32) + i for i in range(length)],
            "action": [np.ones(3, dtype=np.float32) * i for i in range(length)],
            "task_index": np.zeros(length, dtype=np.int64),
        }
    )
    df.to_parquet(path)


class TestParquetMmap:
    def test_builds_and_reloads(self, tmp_path: Path) -> None:
        repo = tmp_path / "dataset"
        parquet = repo / "data" / "chunk-000" / "episode_000000.parquet"
        _write_episode_parquet(parquet)

        store = MmapTrajectoryStore(repo, enabled=True)
        view1 = store.load(parquet)
        view2 = store.load(parquet)

        assert isinstance(view1, TrajectoryDataView)
        assert view1 is view2
        assert view1["action"].to_numpy().shape == (4, 3)
        # Random parquet rows match the original file.
        raw = pd.read_parquet(parquet)
        for i in (3, 0, 2):
            assert view1["action"].to_numpy()[i].tolist() == raw["action"].iloc[i].tolist()
            assert float(view1["timestamp"].to_numpy()[i]) == float(raw["timestamp"].iloc[i])

        manifest = json.loads(
            (repo / ".mmap" / "data__chunk-000__episode_000000" / "manifest.json").read_text()
        )
        assert manifest["source"] == str(parquet)

    def test_episode_rows(self, tmp_path: Path) -> None:
        repo = tmp_path / "dataset"
        parquet = repo / "data" / "chunk-000" / "file-000.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for ep in (0, 1):
            for t in range(3):
                rows.append(
                    {
                        "episode_index": ep,
                        "timestamp": float(t),
                        "observation.state": np.array([ep, t, 0], dtype=np.float32),
                        "action": np.array([ep, t, 1], dtype=np.float32),
                    }
                )
        pd.DataFrame(rows).to_parquet(parquet)

        view = MmapTrajectoryStore(repo).load_episode_rows(parquet, 1)
        assert view["episode_index"].to_numpy().tolist() == [1, 1, 1]
        assert view["action"].to_numpy()[2].tolist() == [1.0, 2.0, 1.0]

    def test_skips_image_bytes_columns(self, tmp_path: Path) -> None:
        repo = tmp_path / "dataset"
        parquet = repo / "data" / "chunk-000" / "episode_000000.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "episode_index": np.zeros(2, dtype=np.int64),
                "timestamp": np.arange(2, dtype=np.float64),
                "image": [{"bytes": b"\x89PNG"} for _ in range(2)],
            }
        ).to_parquet(parquet)
        store = MmapTrajectoryStore(repo, enabled=True)
        view = store.load(parquet)
        assert "image" not in view
        ts = view["timestamp"].to_numpy()
        assert isinstance(ts, np.memmap)
        assert ts.shape == (2,)
        cache = repo / ".mmap" / "data__chunk-000__episode_000000"
        assert not (cache / "image.npy").exists()

    def test_prebuild_files_writes_cache(self, tmp_path: Path) -> None:
        repo = tmp_path / "dataset"
        parquet = repo / "data" / "chunk-000" / "episode_000000.parquet"
        _write_episode_parquet(parquet)
        store = MmapTrajectoryStore(repo, enabled=True)
        built, skipped = store.prebuild_files([parquet])
        assert built == 1 and skipped == 0
        assert (repo / ".mmap" / "data__chunk-000__episode_000000" / "manifest.json").is_file()
        built2, skipped2 = store.prebuild_files([parquet, parquet])
        assert built2 == 0 and skipped2 == 1

    def test_prebuild_skip_does_not_open_npy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = tmp_path / "dataset"
        parquet = repo / "data" / "chunk-000" / "episode_000000.parquet"
        _write_episode_parquet(parquet)
        store = MmapTrajectoryStore(repo, enabled=True)
        assert store.prebuild_files([parquet]) == (1, 0)

        def _boom(*_a, **_k):
            raise AssertionError("skip-check must not mmap column files")

        monkeypatch.setattr("lbm.dataloader.mmap.mmap_io._load_episode_view", _boom)
        assert store.prebuild_files([parquet]) == (0, 1)

    def test_prebuild_files_parallel(self, tmp_path: Path) -> None:
        repo = tmp_path / "dataset"
        paths = []
        for i in range(3):
            parquet = repo / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
            _write_episode_parquet(parquet)
            paths.append(parquet)
        store = MmapTrajectoryStore(repo, enabled=True)
        built, skipped = store.prebuild_files(paths, workers=2)
        assert built == 3 and skipped == 0
        for i in range(3):
            assert (repo / ".mmap" / "data__chunk-000" / f"episode_{i:06d}" / "manifest.json").is_file() or (
                repo / ".mmap" / f"data__chunk-000__episode_{i:06d}" / "manifest.json"
            ).is_file()

    def test_disabled_uses_parquet(self, tmp_path: Path) -> None:
        repo = tmp_path / "dataset"
        parquet = repo / "data" / "chunk-000" / "episode_000001.parquet"
        _write_episode_parquet(parquet, length=2)
        view = MmapTrajectoryStore(repo, enabled=False).load(parquet)
        assert view["timestamp"].to_numpy().shape == (2,)


class TestFrameMmap:
    def test_jpeg_roundtrip(self) -> None:
        frame = np.full((24, 32, 3), 127, dtype=np.uint8)
        out = decode_jpeg_rgb(encode_jpeg_rgb(frame, quality=85))
        assert out.shape == frame.shape

    def test_decode_into_rgb(self) -> None:
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[..., 0] = 200
        blob = encode_jpeg_rgb(rgb, quality=95)
        dest = np.empty((1, 16, 16, 3), dtype=np.uint8)
        decode_jpegs_into([blob], dest, executor=None)
        assert dest[0, 0, 0, 0] > 150
        assert dest[0, 0, 0, 2] < 50

    def test_letterbox(self) -> None:
        from lbm.dataloader.mmap.jpeg_io import letterbox_content_hw

        img = np.zeros((32, 48, 3), dtype=np.uint8)
        square = resize_with_pad(img, 16)
        assert square.shape == (16, 16, 3)
        assert letterbox_content_hw(512, 640, 224) == (179, 224)
        assert letterbox_content_hw(640, 512, 224) == (224, 179)
        assert letterbox_content_hw(224, 224, 224) == (224, 224)
        content = np.zeros((179, 224, 3), dtype=np.uint8)
        content[10, 20] = 200
        padded = resize_with_pad(content, 224)
        assert padded.shape == (224, 224, 3)
        y0 = (224 - 179) // 2
        assert int(padded[y0 + 10, 20, 0]) == 200
        assert int(padded[0, 0, 0]) == 0

    def test_source_jpegs_match_rgb_write(self, tmp_path: Path) -> None:
        rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        rgb[:, :24, 0] = 180
        rgb[:, 24:, 1] = 180
        blob = encode_jpeg_rgb(rgb, quality=95)
        decoded = decode_jpeg_rgb(blob)
        a = tmp_path / "rgb"
        b = tmp_path / "still"
        write_frame_cache(a, source_tag="t", frames=np.stack([decoded]), jpeg_quality=85, image_size=16)
        write_frame_cache(b, source_tag="t", source_jpegs=[blob], jpeg_quality=85, image_size=16)
        ea = MmapFrameEpisode.open(a)
        eb = MmapFrameEpisode.open(b)
        try:
            fa = ea.get_frames(np.array([0]))
            fb = eb.get_frames(np.array([0]))
            assert fa.shape == fb.shape == (1, 16, 16, 3)
            assert np.array_equal(fa, fb)
        finally:
            ea.close()
            eb.close()

    def test_shared_packed_mp4_decodes_file_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from lbm.dataloader.custom.common.lerobot import LerobotCam, LerobotDump

        repo = tmp_path / "dataset"
        video = repo / "videos" / "cam.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"fake")
        parquet = repo / "data.parquet"
        parquet.write_bytes(b"x")

        def _dump(epi: int, origin: float) -> LerobotDump:
            return LerobotDump(
                repo=repo,
                version="v3",
                episode_index=epi,
                parquet=parquet,
                info={},
                video_tmpl="",
                fps=10.0,
                videos={"cam": LerobotCam(video_key="observation.images.cam", from_timestamp=origin)},
            )

        store = MmapFrameStore(repo, jpeg_quality=85, image_size=None, decode_workers=1)
        jobs = [
            store.make_job(
                trajectory_id=epi,
                video_key="cam",
                video_path=video,
                episode_timestamps=None,
                from_timestamp=0.0,
                video_backend="custom",
                video_backend_kwargs={
                    "mode": "lerobot",
                    "cam": "cam",
                    "n_frames": 4,
                    "extra": {"lerobot": _dump(epi, origin)},
                },
            )
            for epi, origin in ((0, 0.0), (1, 0.4))
        ]
        try:
            assert lerobot_v3_file_span(jobs[0]) == (0, 4)
            assert lerobot_v3_file_span(jobs[1]) == (4, 8)
            shared, solo = partition_shared_frame_jobs(jobs)
            assert len(shared) == 1 and shared[0] == jobs
            assert solo == []

            calls = {"n": 0}
            frames = [np.full((8, 8, 3), i * 10, dtype=np.uint8) for i in range(8)]

            def _iter(_path, start, stop, **_k):
                calls["n"] += 1
                assert (start, stop) == (0, 8)
                yield from frames[start:stop]

            monkeypatch.setattr("lbm.dataloader.custom.video.iter_mp4_span", _iter)
            write_shared_mp4_caches(store, jobs, progress=False)
            assert calls["n"] == 1
            e0 = MmapFrameEpisode.open(store.cache_dir_for(0, "cam"))
            e1 = MmapFrameEpisode.open(store.cache_dir_for(1, "cam"))
            try:
                assert int(e0.get_frames(np.array([0]))[0, 0, 0, 0]) == 0
                assert int(e1.get_frames(np.array([0]))[0, 0, 0, 0]) == 40
            finally:
                e0.close()
                e1.close()

            import shutil

            shutil.rmtree(store.cache_root)
            store._episodes.clear()
            calls["n"] = 0
            built, skipped = store.prebuild(jobs, workers=1)
            assert built == 2 and skipped == 0
            assert calls["n"] == 1
        finally:
            store.close()

    def test_pack_and_read(self, tmp_path: Path) -> None:
        frames = np.stack(
            [np.full((16, 16, 3), fill_value=i, dtype=np.uint8) for i in range(5)],
            axis=0,
        )
        cache_dir = tmp_path / "cache"
        write_frame_cache(cache_dir, source_tag="test", frames=frames, jpeg_quality=85)
        episode = MmapFrameEpisode.open(cache_dir)
        try:
            out = episode.get_frames(np.array([0, 2, 4]))
            assert out.shape == (3, 16, 16, 3)
            assert out[1][0, 0, 0] == 2
            shuffled = episode.get_frames(np.array([4, 0, 2]))
            assert shuffled[0][0, 0, 0] == 4
            assert shuffled[1][0, 0, 0] == 0
            assert shuffled[2][0, 0, 0] == 2
        finally:
            episode.close()

    def test_store_build(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "chunk-000" / "cam" / "episode_000000.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        frames = np.stack(
            [np.full((8, 8, 3), fill_value=i, dtype=np.uint8) for i in range(4)],
            axis=0,
        )
        monkeypatch.setattr(
            "lbm.dataloader.mmap.frame_mmap_io.get_all_frames",
            lambda *a, **k: frames,
        )
        store = MmapFrameStore(repo, jpeg_quality=85, image_size=None, decode_workers=1)
        out = store.get_frames(
            trajectory_id=0,
            video_key="cam",
            video_path=video_path,
            frame_indices=np.array([1, 3]),
            episode_timestamps=None,
            from_timestamp=0.0,
            video_backend="decord",
            video_backend_kwargs={},
        )
        assert out.shape == (2, 8, 8, 3)
        assert (repo / ".mmap" / "frames" / "episode_000000" / "cam" / "frames.bin").is_file()

    def test_rebuilds_empty_manifest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "cam.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        frames = np.stack([np.full((8, 8, 3), i, dtype=np.uint8) for i in range(3)], axis=0)
        monkeypatch.setattr(
            "lbm.dataloader.mmap.frame_mmap_io.get_all_frames",
            lambda *a, **k: frames,
        )
        cache_dir = repo / ".mmap" / "frames" / "episode_000000" / "cam"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "manifest.json").write_text("", encoding="utf-8")

        store = MmapFrameStore(repo, jpeg_quality=85, image_size=None, decode_workers=1)
        out = store.get_frames(
            trajectory_id=0,
            video_key="cam",
            video_path=video_path,
            frame_indices=np.array([0, 2]),
            episode_timestamps=None,
            from_timestamp=0.0,
            video_backend="decord",
            video_backend_kwargs={},
        )
        assert out.shape == (2, 8, 8, 3)
        assert json.loads((cache_dir / "manifest.json").read_text())["layout"] == "jpeg_pack"

    def test_letterbox_jpeg_smaller_than_raw(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        yy, xx = np.mgrid[0:48, 0:64]
        frame = np.stack([xx * 4, yy * 5, np.full_like(xx, 80)], axis=-1).astype(np.uint8)
        frames = np.stack([frame] * 4, axis=0)
        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "cam.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        monkeypatch.setattr(
            "lbm.dataloader.mmap.frame_mmap_io.get_all_frames",
            lambda *a, **k: frames,
        )
        store = MmapFrameStore(repo, jpeg_quality=85, image_size=16, decode_workers=1)
        out = store.get_frames(
            trajectory_id=0,
            video_key="cam",
            video_path=video_path,
            frame_indices=np.array([0, 1]),
            episode_timestamps=None,
            from_timestamp=0.0,
            video_backend="decord",
            video_backend_kwargs={},
        )
        assert out.shape == (2, 16, 16, 3)
        blob_size = (repo / ".mmap" / "frames" / "episode_000000" / "cam" / "frames.bin").stat().st_size
        native_raw = int(np.prod(frames.shape))
        assert blob_size < native_raw // 8
        manifest = json.loads(
            (repo / ".mmap" / "frames" / "episode_000000" / "cam" / "manifest.json").read_text()
        )
        assert manifest["layout"] == "jpeg_pack"
        assert manifest["jpeg_quality"] == 85

    def test_pack_layout(self) -> None:
        blob, offset, length = pack_jpeg_frames([b"abc", b"de", b"fghi"])
        assert blob == b"abcdefghi"
        assert offset.tolist() == [0, 3, 5]
        assert length.tolist() == [3, 2, 4]

    def test_align_frames_to_parquet_steps(self) -> None:
        frames = np.stack([np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)], axis=0)
        # Same fps, extra decoded frames: keep one slot per parquet step (0..7).
        timestamps = np.arange(8, dtype=np.float64) / 5.0
        aligned = align_frames_to_parquet_steps(frames, timestamps)
        assert aligned.shape[0] == 8
        assert [int(aligned[i, 0, 0, 0]) for i in range(8)] == list(range(8))
        subset = frames[:8]
        same = align_frames_to_parquet_steps(subset, timestamps)
        assert same is subset

    def test_cache_slots_follow_parquet_timestamps(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        frames = np.stack([np.full((8, 8, 3), i * 20, dtype=np.uint8) for i in range(10)], axis=0)
        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "cam.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        monkeypatch.setattr(
            "lbm.dataloader.mmap.frame_mmap_io.get_all_frames",
            lambda *a, **k: frames,
        )
        store = MmapFrameStore(repo, jpeg_quality=95, image_size=None, decode_workers=1)
        try:
            timestamps = np.arange(8, dtype=np.float64) / 5.0
            # Shuffled parquet steps; cache has 8 slots even though the mp4 decoded 10 frames.
            out = store.get_frames(
                trajectory_id=0,
                video_key="cam",
                video_path=video_path,
                frame_indices=np.array([7, 0, 3]),
                episode_timestamps=timestamps,
                from_timestamp=0.0,
                video_backend="decord",
                video_backend_kwargs={},
            )
            assert out.shape == (3, 8, 8, 3)
            assert int(out[0, 0, 0, 0]) == 140
            assert int(out[1, 0, 0, 0]) == 0
            assert int(out[2, 0, 0, 0]) == 60
            manifest = json.loads(
                (repo / ".mmap" / "frames" / "episode_000000" / "cam" / "manifest.json").read_text()
            )
            assert manifest["num_frames"] == 8
        finally:
            store.close()

    def test_source_tag_change_rebuilds_from_video(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        frames = np.stack([np.full((8, 8, 3), 40, dtype=np.uint8) for _ in range(3)], axis=0)
        calls = {"n": 0}

        def _frames(*_a, **_k):
            calls["n"] += 1
            return frames

        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "cam.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        monkeypatch.setattr("lbm.dataloader.mmap.frame_mmap_io.get_all_frames", _frames)
        kwargs = dict(
            trajectory_id=0,
            video_key="cam",
            video_path=video_path,
            frame_indices=np.array([0]),
            episode_timestamps=None,
            from_timestamp=0.0,
            video_backend="decord",
            video_backend_kwargs={},
        )
        store85 = MmapFrameStore(repo, jpeg_quality=85, image_size=None, decode_workers=1)
        store90 = MmapFrameStore(repo, jpeg_quality=90, image_size=None, decode_workers=1)
        try:
            store85.get_frames(**kwargs)
            store90.get_frames(**kwargs)
            assert calls["n"] == 2
        finally:
            store85.close()
            store90.close()
        store = MmapFrameStore(tmp_path, decode_workers=4)
        try:
            assert store._decode_pool is None
            assert store._decode_pool_pid is None
        finally:
            store.close()

    def test_prebuild_then_train_does_not_decode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "cam.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        frames = np.stack([np.full((8, 8, 3), i, dtype=np.uint8) for i in range(4)], axis=0)
        calls = {"n": 0}

        def _frames(*_a, **_k):
            calls["n"] += 1
            return frames

        monkeypatch.setattr("lbm.dataloader.mmap.frame_mmap_io.get_all_frames", _frames)
        store = MmapFrameStore(repo, jpeg_quality=85, image_size=None, decode_workers=1)
        job = store.make_job(
            trajectory_id=0,
            video_key="cam",
            video_path=video_path,
            episode_timestamps=None,
            from_timestamp=0.0,
            video_backend="decord",
            video_backend_kwargs={},
        )
        try:
            built, skipped = store.prebuild([job], workers=1)
            assert built == 1 and skipped == 0
            assert calls["n"] == 1
            store.allow_build = False
            out = store.get_frames(
                trajectory_id=0,
                video_key="cam",
                video_path=video_path,
                frame_indices=np.array([1, 3]),
                episode_timestamps=None,
                from_timestamp=0.0,
                video_backend="decord",
                video_backend_kwargs={},
            )
            assert out.shape == (2, 8, 8, 3)
            assert calls["n"] == 1
            built2, skipped2 = store.prebuild([job], workers=1)
            assert built2 == 0 and skipped2 == 1
        finally:
            store.close()

    def test_job_ready_skips_source_stat_when_cache_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "cam.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        store = MmapFrameStore(repo, jpeg_quality=85, image_size=None, decode_workers=1)
        job = store.make_job(
            trajectory_id=0,
            video_key="cam",
            video_path=video_path,
            episode_timestamps=None,
            from_timestamp=0.0,
            video_backend="decord",
            video_backend_kwargs={},
        )
        try:

            def _boom(*_a, **_k):
                raise AssertionError("source_tag should not run until cache files exist")

            monkeypatch.setattr(store, "_source_tag", _boom)
            assert store.job_ready(job) is False
            job["video_path"] = str(tmp_path / "missing.mp4")
            assert store.job_ready(job) is False
        finally:
            store.close()

    def test_split_pending_skips_ready_when_root_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "cam.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        store = MmapFrameStore(repo, jpeg_quality=85, image_size=None, decode_workers=1)
        job = store.make_job(
            trajectory_id=0,
            video_key="cam",
            video_path=video_path,
            episode_timestamps=None,
            from_timestamp=0.0,
            video_backend="decord",
            video_backend_kwargs={},
        )
        try:

            def _boom(_job: dict) -> bool:
                raise AssertionError("job_ready should not run when cache_root is missing")

            monkeypatch.setattr(store, "job_ready", _boom)
            pending, skipped = split_pending_frame_jobs({str(store.dataset_path): (store, [job, job])})
            assert skipped == 0
            assert len(next(iter(pending.values()))[1]) == 2
        finally:
            store.close()

    def test_allow_build_false_raises_if_missing(self, tmp_path: Path) -> None:
        repo = tmp_path / "dataset"
        video_path = repo / "videos" / "cam.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
        store = MmapFrameStore(repo, jpeg_quality=85, image_size=None, decode_workers=1)
        store.allow_build = False
        try:
            with pytest.raises(FileNotFoundError, match="mmap frame cache missing"):
                store.get_frames(
                    trajectory_id=0,
                    video_key="cam",
                    video_path=video_path,
                    frame_indices=np.array([0]),
                    episode_timestamps=None,
                    from_timestamp=0.0,
                    video_backend="decord",
                    video_backend_kwargs={},
                )
        finally:
            store.close()

    def test_lazy_pool_runs_task(self, tmp_path: Path) -> None:
        store = MmapFrameStore(tmp_path, decode_workers=2)
        try:
            pool = store._get_decode_pool()
            assert pool is not None
            assert pool.submit(lambda: 41).result(timeout=5) == 41
        finally:
            store.close()

    def test_thread_pool_jpeg_decode_completes(self) -> None:
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[..., 1] = 180
        blob = encode_jpeg_rgb(rgb, quality=95)
        dest = np.empty((8, 16, 16, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=2) as pool:
            decode_jpegs_into([blob] * 8, dest, pool)
        assert dest[0, 0, 0, 1] > 100

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
    def test_decode_pool_works_after_fork(self, tmp_path: Path) -> None:
        store = MmapFrameStore(tmp_path, decode_workers=2)
        proc = None
        try:
            assert store._decode_pool is None

            def _child() -> None:
                pool = store._get_decode_pool()
                if pool is None:
                    raise SystemExit(2)
                if pool.submit(lambda: 7).result(timeout=5) != 7:
                    raise SystemExit(3)

            ctx = multiprocessing.get_context("fork")
            proc = ctx.Process(target=_child)
            proc.start()
            proc.join(timeout=15)
            assert proc.exitcode == 0, f"child exitcode={proc.exitcode}"
        finally:
            if proc is not None and proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            store.close()


    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
    def test_decode_pool_recreated_when_parent_had_threads(self, tmp_path: Path) -> None:
        """DataLoader fork after the parent already ran JPEG decode must not hang."""
        store = MmapFrameStore(tmp_path, decode_workers=2)
        proc = None
        try:
            parent_pool = store._get_decode_pool()
            assert parent_pool is not None
            assert parent_pool.submit(lambda: 1).result(timeout=5) == 1

            def _child() -> None:
                pool = store._get_decode_pool()
                if pool is None:
                    raise SystemExit(2)
                if pool is parent_pool:
                    raise SystemExit(4)
                if pool.submit(lambda: 9).result(timeout=5) != 9:
                    raise SystemExit(3)

            ctx = multiprocessing.get_context("fork")
            proc = ctx.Process(target=_child)
            proc.start()
            proc.join(timeout=15)
            assert proc.exitcode == 0, f"child exitcode={proc.exitcode}"
        finally:
            if proc is not None and proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            store.close()


def test_collate_uses_uint8_tensor() -> None:
    from lbm.dataloader.pad import collate_fn

    batch = [
        {
            "image": [np.zeros((2, 8, 8, 3), dtype=np.uint8), np.ones((2, 8, 8, 3), dtype=np.uint8)],
            "action": np.zeros((16, 4), dtype=np.float16),
            "lang": "pick",
            "robot_tag": "new_embodiment",
            "camera_keys": ("cam_a", "cam_b"),
        }
        for _ in range(3)
    ]
    out = collate_fn(batch)
    assert tuple(out["image"].shape) == (3, 2, 2, 8, 8, 3)
    assert out["image"].dtype == torch.uint8
    assert int(out["image"][0, 1, 0, 0, 0, 0]) == 1
    assert tuple(out["action"].shape) == (3, 16, 4)
    assert tuple(out["embodiment_id"].shape) == (3,)


def test_collate_passthrough_prebatched() -> None:
    from lbm.dataloader.pad import collate_fn

    image = torch.zeros((2, 3, 4, 8, 8, 3), dtype=torch.uint8)
    batched = {
        "image": image,
        "action": torch.zeros((2, 16, 4)),
        "lang": ["a", "b"],
        "robot_tag": ["new_embodiment", "new_embodiment"],
    }
    out = collate_fn(batched)
    assert out is batched
    assert tuple(out["image"].shape) == (2, 3, 4, 8, 8, 3)


def test_decode_threads_for_loaders() -> None:
    from lbm.dataloader.mmap.jpeg_io import decode_threads_for_loaders

    assert decode_threads_for_loaders(16, cpus=64) == 4
    assert decode_threads_for_loaders(16, cpus=256) == 4
    assert decode_threads_for_loaders(16, cpus=8) == 1
    assert decode_threads_for_loaders(1, cpus=64) == 8
    assert decode_threads_for_loaders(2, cpus=64) == 8
