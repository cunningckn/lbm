"""Official download → LBM dump converters (not scan / FK / mmap)."""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_SCRIPTS = ROOT / "scripts" / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from catalog import DUMPS, NAMES  # noqa: E402
from converters import agibot as agibot_c  # noqa: E402
from converters import das as das_c  # noqa: E402
from converters import detect  # noqa: E402
from converters import galaxea as galaxea_c  # noqa: E402
from converters import rmbench as rmbench_c  # noqa: E402
from process import process_one  # noqa: E402


def test_catalog_covers_mix_names():
    from lbm.dataloader.custom.datasets.mixes import ALL

    assert tuple(NAMES) == ALL
    for name, spec in DUMPS.items():
        assert spec["urls"], name
        assert spec["urls"][0].startswith("http"), name
        assert spec["process"] in {"none", "galaxea_extract", "agibot_layout", "rmbench_hdf5", "das_slim"}


def test_catalog_bash_emits_repo():
    from catalog import emit_bash

    text = emit_bash("droid")
    assert "lerobot/droid_1.0.1" in text
    assert "https://huggingface.co/datasets/lerobot/droid_1.0.1" in text
    text = emit_bash("agibot")
    assert "agibot-world/AgiBotWorld-Alpha" in text
    assert "agibot-world/AgiBotWorld-Beta" in text
    text = emit_bash("galaxea")
    assert "OpenGalaxea/Galaxea-Open-World-Dataset" in text
    assert "lerobot" in text


def _info(path: Path, task: str = "demo") -> None:
    meta = path / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.1", "fps": 15, "task": task}))


def test_galaxea_extracts_task_tarball(tmp_path: Path):
    raw = tmp_path / "raw"
    dest = tmp_path / "dest"
    task = "Clean_The_Mirror_20250714_006"
    bundle = tmp_path / "bundle"
    _info(bundle / task)
    tar_path = raw / "lerobot" / f"{task}.tar.gz"
    tar_path.parent.mkdir(parents=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(bundle / task, arcname=task)
    galaxea_c.extract_galaxea(raw, dest)
    assert (dest / task / "meta" / "info.json").is_file()
    assert detect.ready_kind(dest, "galaxea") == "galaxea"


def test_galaxea_flattens_nested_task_dir(tmp_path: Path):
    raw = tmp_path / "raw"
    dest = tmp_path / "dest"
    task = "Arrange_Fruits_20250819_011"
    inner = tmp_path / "bundle" / task / task
    _info(inner)
    tar_path = raw / f"{task}.tar.gz"
    raw.mkdir()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(tmp_path / "bundle" / task, arcname=task)
    galaxea_c.extract_galaxea(raw, dest)
    assert (dest / task / "meta" / "info.json").is_file()
    assert not (dest / task / task / "meta" / "info.json").exists()


def test_agibot_nests_alpha_beta(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "alpha" / "proprio_stats").mkdir(parents=True)
    (raw / "beta" / "proprio_stats").mkdir(parents=True)
    dest = tmp_path / "dest"
    agibot_c.layout_agibot(raw, dest)
    assert (dest / "AgiBotWorld_alpha" / "proprio_stats").is_dir()
    assert (dest / "AgiBotWorld_beta" / "proprio_stats").is_dir()
    assert detect.ready_kind(dest, "agibot") == "agibot"


def test_rmbench_hdf5_to_lerobot(tmp_path: Path):
    pytest.importorskip("h5py")
    pytest.importorskip("cv2")
    import h5py

    raw = tmp_path / "raw"
    h5 = raw / "data" / "cover_blocks" / "demo_clean" / "data" / "episode0.hdf5"
    h5.parent.mkdir(parents=True)
    t, h, w = 4, 16, 16
    with h5py.File(h5, "w") as f:
        f.create_dataset("observations/qpos", data=np.zeros((t, 14), np.float32))
        f.create_dataset("action", data=np.ones((t, 14), np.float32))
        imgs = f.create_group("observations/images")
        imgs.create_dataset("cam_high", data=np.zeros((t, h, w, 3), np.uint8))
        imgs.create_dataset("cam_left_wrist", data=np.zeros((t, h, w, 3), np.uint8))
        imgs.create_dataset("cam_right_wrist", data=np.zeros((t, h, w, 3), np.uint8))
    dest = tmp_path / "dest"
    rmbench_c.convert_rmbench(raw, dest, fps=10.0)
    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 1
    assert (dest / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert detect.ready_kind(dest, "rmbench") == "lerobot"


def test_das_official_hdf5_to_slim(tmp_path: Path):
    pytest.importorskip("h5py")
    pytest.importorskip("cv2")
    import h5py

    raw = tmp_path / "raw"
    src = raw / "ep0.h5"
    raw.mkdir()
    t, h, w = 3, 8, 8
    with h5py.File(src, "w") as f:
        f.create_dataset("observations/eef_pos", data=np.linspace(0, 1, t * 8).reshape(t, 8))
        cams = f.create_group("observations/cameras")
        cams.create_dataset("left_wrist", data=np.zeros((t, h, w, 3), np.uint8) + 10)
        cams.create_dataset("right_wrist", data=np.zeros((t, h, w, 3), np.uint8) + 20)
    dest = tmp_path / "dest"
    das_c.convert_das(raw, dest)
    assert detect.ready_kind(dest, "das_gripper") == "das_slim"
    meta = json.loads((dest / "sample" / "das_gripper_slim_meta.json").read_text())
    assert meta["episodes"]


def test_process_none_symlinks_raw(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw" / "droid"
    dest = tmp_path / "datasets" / "droid"
    raw.mkdir(parents=True)
    (raw / "meta").mkdir()
    (raw / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
    monkeypatch.setenv("LBM_DATASETS", str(tmp_path / "datasets"))
    monkeypatch.setenv("LBM_DATA_RAW", str(tmp_path / "raw"))
    process_one("droid", src=raw, dest=dest)
    assert dest.exists()
    assert (dest / "meta" / "info.json").is_file()


def test_process_skips_ready_dest(tmp_path: Path):
    dest = tmp_path / "abc"
    (dest / "data" / "train" / "task=fold" / "episode_0000").mkdir(parents=True)
    (dest / "data" / "train" / "task=fold" / "episode_0000" / "episode.mcap").write_bytes(b"mcap")
    assert detect.ready_kind(dest, "abc") == "mcap"
    out = process_one("abc", src=tmp_path / "missing", dest=dest)
    assert out == dest
