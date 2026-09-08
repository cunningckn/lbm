from pathlib import Path

import pytest

from lbm import (
    ClipConfig,
    DiTConfig,
    TrainConfig,
    default_checkpoints_dir,
    find_lerobot_dataset,
    temporal_summary,
    validate_model_config,
)
from lbm.train_loop import main


def test_train_config_valid():
    cfg = TrainConfig(fake_data=True, train_steps=1, batch_size=1)
    assert validate_model_config(cfg.model) == []
    assert "action=" in temporal_summary(cfg.model)


def test_train_loop_importable():
    assert callable(main)
    assert DiTConfig().chunk_length == 50


def test_find_lerobot_dataset_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_lerobot_dataset("no_such_dataset") == ""


def test_find_lerobot_dataset_discovers_tree(tmp_path, monkeypatch):
    root = tmp_path / "datasets" / "demo"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    path = find_lerobot_dataset("demo")
    assert path.endswith("datasets/demo")
    assert (Path(path) / "meta" / "info.json").is_file()


@pytest.mark.integration
def test_rmbench_on_disk_if_present():
    path = find_lerobot_dataset("rmbench")
    if not path:
        pytest.skip("rmbench not on disk")
    assert (Path(path) / "meta" / "info.json").is_file()


def test_default_checkpoints_dir():
    root = default_checkpoints_dir()
    assert root.name == "checkpoints"
    assert (root.parent / "src" / "lbm").is_dir()
    assert Path(ClipConfig().cache_dir) == root / "clip"
    assert Path(TrainConfig().output_dir) == root
