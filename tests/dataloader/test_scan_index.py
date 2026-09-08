"""Episode index cache at ``{root}/.cache/episodes``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lbm.dataloader.custom.common.lerobot import LerobotCam, LerobotDump
from lbm.dataloader.custom.datasets import CUSTOM_SPECS
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.scan import scan_root
from lbm.dataloader.custom.scan_index import (
    SCANNER_VERSION,
    ScanIndexError,
    cache_dir,
    load_scan_index,
    save_scan_index,
)

DAS = CUSTOM_SPECS["das_gripper"]


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _das_records(root: Path, n: int = 2) -> list[EpisodeRecord]:
    recs = []
    for i in range(n):
        h5 = _touch(root / f"ep{i}" / "episode.hdf5")
        recs.append(
            EpisodeRecord(
                kind="das",
                path=str(h5),
                n_frames=10 + i,
                lang=f"task {i}",
                extra={"dir": str(h5.parent), "videos": {"cam_left_wrist": str(h5.parent / "left.mp4")}},
            )
        )
    return recs


def _dump(repo: Path, parquet: Path, epi: int) -> LerobotDump:
    return LerobotDump(
        repo=repo,
        version="v2",
        episode_index=epi,
        parquet=parquet,
        info={"codebase_version": "v2.1", "fps": 30.0},
        video_tmpl="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        fps=30.0,
        chunk=0,
        videos={
            "cam_high": LerobotCam(
                video_key="observation.images.cam_high", chunk=0, file_index=0, from_timestamp=0.0
            )
        },
    )


def _stub_scan(monkeypatch: pytest.MonkeyPatch, recs: list[EpisodeRecord]) -> dict:
    calls = {"n": 0}

    def fake_scan(root, spec, *, max_episodes=None):
        calls["n"] += 1
        return recs if max_episodes is None else recs[:max_episodes]

    monkeypatch.setattr("lbm.dataloader.custom.datasets.das_gripper.scan", fake_scan)
    return calls


def test_roundtrip_writes_manifest_repos_episodes(tmp_path: Path):
    records = _das_records(tmp_path)
    dest = save_scan_index(tmp_path, DAS, records)
    assert dest == tmp_path / ".cache" / "episodes"
    man = json.loads((dest / "manifest.json").read_text())
    assert man["scanner_version"] == SCANNER_VERSION
    assert man["complete"] is True
    assert man["n_records"] == 2
    assert man["n_steps"] == 21
    assert man["spec"]["name"] == "das_gripper"
    assert (dest / "repos.jsonl").read_text() == ""
    loaded = load_scan_index(tmp_path, DAS)
    assert [r.path for r in loaded] == [r.path for r in records]
    assert loaded[0].n_frames == 10
    assert loaded[0].extra["videos"]["cam_left_wrist"].endswith("left.mp4")
    assert loaded[1].lang == "task 1"


def test_lerobot_interns_one_repo_line(tmp_path: Path):
    spec = CUSTOM_SPECS["rmbench"]
    repo = tmp_path / "repo"
    p0 = _touch(repo / "data" / "episode_000000.parquet")
    p1 = _touch(repo / "data" / "episode_000001.parquet")
    records = [
        EpisodeRecord(kind="lerobot_v2", path=str(p0), n_frames=4, extra={"lerobot": _dump(repo, p0, 0)}),
        EpisodeRecord(kind="lerobot_v2", path=str(p1), n_frames=5, extra={"lerobot": _dump(repo, p1, 1)}),
    ]
    dest = save_scan_index(tmp_path, spec, records)
    lines = [ln for ln in (dest / "repos.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["repo"] == str(repo)
    loaded = load_scan_index(tmp_path, spec)
    d0, d1 = loaded[0].extra["lerobot"], loaded[1].extra["lerobot"]
    assert (d0.episode_index, d1.episode_index) == (0, 1)
    assert d0.info == d1.info
    assert d0.videos["cam_high"].video_key == "observation.images.cam_high"
    assert d1.parquet == p1


def test_scan_root_reuses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    recs = _das_records(tmp_path)
    calls = _stub_scan(monkeypatch, recs)
    first = scan_root(tmp_path, DAS)
    second = scan_root(tmp_path, DAS)
    assert calls["n"] == 1
    assert (cache_dir(tmp_path) / "manifest.json").is_file()
    assert [r.path for r in second] == [r.path for r in first]


def test_max_episodes_does_not_write_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    recs = _das_records(tmp_path)
    _stub_scan(monkeypatch, recs)
    out = scan_root(tmp_path, DAS, max_episodes=1)
    assert len(out) == 1
    assert not cache_dir(tmp_path).exists()


def test_incomplete_cache_raises(tmp_path: Path):
    save_scan_index(tmp_path, DAS, _das_records(tmp_path))
    (cache_dir(tmp_path) / "episodes.parquet").unlink()
    with pytest.raises(ScanIndexError, match="incomplete"):
        load_scan_index(tmp_path, DAS)
    with pytest.raises(ScanIndexError, match="incomplete"):
        scan_root(tmp_path, DAS)


def test_spec_mismatch_raises(tmp_path: Path):
    save_scan_index(tmp_path, DAS, _das_records(tmp_path))
    with pytest.raises(ScanIndexError, match="spec "):
        load_scan_index(tmp_path, CUSTOM_SPECS["kai0"])


def test_rescan_rebuilds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    recs = _das_records(tmp_path)
    calls = _stub_scan(monkeypatch, recs)
    scan_root(tmp_path, DAS)
    scan_root(tmp_path, DAS, rescan=True)
    assert calls["n"] == 2


def test_refuses_empty_write(tmp_path: Path):
    with pytest.raises(ScanIndexError, match="empty"):
        save_scan_index(tmp_path, DAS, [])
