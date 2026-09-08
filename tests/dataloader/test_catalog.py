"""Named datasets under ``lbm/datasets/<name>``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lbm.action_space import ABS
from lbm.dataloader.catalog import DATASETS, backend_for, lookup
from lbm.dataloader.custom.datasets import CUSTOM_SPECS
from lbm.dataloader.paths import datasets_root, resolve_dataset


def test_catalog_covers_custom_specs():
    for name in CUSTOM_SPECS:
        entry = lookup(name)
        assert entry is not None
        assert entry.backend == "custom"
        assert entry.robot_type == name
    assert lookup("libero").backend == "custom"
    assert lookup("rmbench").robot_type == "rmbench"
    assert lookup("/data/kai0").name == "kai0"
    assert lookup("robotwin").backend == "custom"
    assert lookup("robotwin").robot_type == "robotwin"
    assert lookup("robodojo") is None


def test_backend_for_named_and_mix():
    assert backend_for(dataset="kai0") == "custom"
    assert backend_for(robot_type="galaxea") == "custom"
    assert backend_for(dataset="libero") == "custom"
    assert backend_for(dataset="rmbench") == "custom"
    assert backend_for(data_mix="kai0") == "custom"
    assert backend_for(data_mix="libero") == "custom"
    assert backend_for(data_mix="robotwin") == "custom"
    assert backend_for(dataset="robotwin") == "custom"
    assert backend_for(data_mix="custom_all") == "custom"
    assert backend_for(data_mix="all") == "custom"
    assert backend_for(data_mix="kai0,galaxea") == "custom"
    assert backend_for(data_mix="kai0,libero") == "custom"


def test_resolve_named_dataset_if_linked():
    root = datasets_root()
    if not (root / "kai0").exists():
        pytest.skip("datasets/kai0 missing")
    path = resolve_dataset("kai0")
    assert path == (root / "kai0").resolve()
    assert resolve_dataset(str(path)) == path


def test_resolve_missing_raises():
    with pytest.raises(FileNotFoundError):
        resolve_dataset("no_such_lbm_dataset_xyz")
    assert resolve_dataset("no_such_lbm_dataset_xyz", required=False) is None


def test_resolve_existing_tmp_path(tmp_path: Path):
    assert resolve_dataset(tmp_path) == tmp_path.resolve()


def test_datasets_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LBM_DATASETS", str(tmp_path))
    assert datasets_root() == tmp_path.resolve()


def test_catalog_folder_names():
    assert set(DATASETS) >= {
        "agibot",
        "galaxea",
        "kai0",
        "egoverse",
        "das_gripper",
        "droid",
        "hy_lance",
        "hifi_umi",
        "abc",
        "libero",
        "rmbench",
        "robotwin",
    }
    assert "robodojo" not in DATASETS


def test_train_loop_routes_named_datasets():
    assert backend_for(dataset="kai0") == "custom"
    assert backend_for(dataset="rmbench") == "custom"
    assert backend_for(data_mix="robotwin") == "custom"
    assert backend_for(data_mix="libero") == "custom"
    assert backend_for(data_mix="all") == "custom"
    assert backend_for(data_mix="custom_all") == "custom"


def test_spec_comes_from_catalog_name_not_info_json(tmp_path: Path, monkeypatch):
    from lbm.config import TrainConfig
    from lbm.dataloader.custom import CUSTOM_SPECS, CustomSingleDataset, Episode
    from lbm.dataloader.mixture import load_named_or_mix

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"robot_type": "Franka"}')
    seen: list[str] = []

    def _fake_make(path, spec_name, data_cfg):
        seen.append(spec_name)
        spec = CUSTOM_SPECS[spec_name]
        n = 2
        return CustomSingleDataset(
            spec,
            episodes=[
                Episode(
                    images={cam: np.zeros((n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys},
                    state=np.zeros((n, spec.state_dim), dtype=np.float32),
                    action=np.zeros((n, spec.action_dim), dtype=np.float32),
                )
            ],
            action_length=0.2,
            action_mode=ABS,
        )

    monkeypatch.setattr("lbm.dataloader.custom.make_custom_dataset", _fake_make)
    cfg = TrainConfig()
    cfg.data.dataset = str(tmp_path)
    with pytest.raises(ValueError, match="unknown spec"):
        load_named_or_mix(cfg)
    assert seen == []
    cfg.data.robot_type = "droid"
    mix = load_named_or_mix(cfg)
    assert seen == ["droid"]
    assert mix.datasets[0].spec.name == "droid"


def test_skip_missing_mix_prints(tmp_path, monkeypatch, capsys):
    from lbm.config import TrainConfig
    from lbm.dataloader.catalog import DatasetEntry
    from lbm.dataloader.custom import CUSTOM_SPECS, CustomSingleDataset, Episode
    from lbm.dataloader.mixture import build_catalog_mixture

    monkeypatch.setenv("LBM_DATASETS", str(tmp_path))
    (tmp_path / "kai0").mkdir()

    def _fake_make(path, spec_name, data_cfg):
        spec = CUSTOM_SPECS[spec_name]
        n = 2
        return CustomSingleDataset(
            spec,
            episodes=[
                Episode(
                    images={cam: np.zeros((n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys},
                    state=np.zeros((n, spec.state_dim), dtype=np.float32),
                    action=np.zeros((n, spec.action_dim), dtype=np.float32),
                )
            ],
            action_length=0.2,
            action_mode=ABS,
        )

    monkeypatch.setattr("lbm.dataloader.custom.make_custom_dataset", _fake_make)
    mix = build_catalog_mixture(
        [DatasetEntry(name="kai0"), DatasetEntry(name="robotwin")],
        TrainConfig(),
        mode="train",
        skip_missing=True,
    )
    assert len(mix.datasets) == 1
    out = capsys.readouterr().out
    assert "skip robotwin" in out
    assert "loaded 1/2" in out


def test_parse_mix_all_and_comma():
    from lbm.dataloader.catalog import parse_mix
    from lbm.dataloader.custom.datasets.mixes import ALL

    assert parse_mix("") is None
    assert parse_mix("libero_all") is None
    plan = parse_mix("robotwin")
    assert plan is not None and plan[0].backend == "custom" and plan[0].name == "robotwin"
    names = [e.name for e in parse_mix("custom_all")]
    assert names == list(ALL)
    all_names = [e.name for e in parse_mix("all")]
    assert all_names == list(ALL)
    assert all_names.count("robotwin") == 1
    pair = parse_mix("kai0,galaxea")
    assert [e.name for e in pair] == ["kai0", "galaxea"]
    mixed = parse_mix("kai0,libero")
    assert [e.backend for e in mixed] == ["custom", "custom"]
    with pytest.raises(KeyError):
        parse_mix("kai0,not_a_dataset")


def test_select_dumps_default_and_comma():
    from lbm.dataloader.catalog import select_dumps

    default = ("kai0", "libero")
    assert select_dumps("", default=default) == default
    assert select_dumps("kai0") == ("kai0",)
    assert select_dumps("kai0,libero") == ("kai0", "libero")
    with pytest.raises(ValueError, match="empty dump list"):
        select_dumps("")
    with pytest.raises(KeyError, match="unknown dump"):
        select_dumps("kai0,not_a_dataset")


def test_on_disk_dumps_skips_missing(tmp_path: Path, capsys):
    from lbm.dataloader.catalog import on_disk_dumps

    (tmp_path / "kai0").mkdir()
    found = on_disk_dumps(("kai0", "libero"), base=tmp_path)
    assert found == [("kai0", (tmp_path / "kai0").resolve())]
    assert "skip libero" in capsys.readouterr().out


def test_load_selected_dumps_uses_catalog_name_as_spec(tmp_path, monkeypatch):
    from lbm.config import TrainConfig
    from lbm.dataloader.custom import CUSTOM_SPECS, CustomSingleDataset, Episode
    from lbm.dataloader.mixture import load_selected_dumps

    target = tmp_path / "aria"
    target.mkdir()
    (tmp_path / "egoverse").symlink_to(target)
    seen: list[str] = []

    def _fake_make(path, spec_name, data_cfg):
        seen.append(spec_name)
        spec = CUSTOM_SPECS[spec_name]
        n = 2
        return CustomSingleDataset(
            spec,
            episodes=[
                Episode(
                    images={cam: np.zeros((n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys},
                    state=np.zeros((n, spec.state_dim), dtype=np.float32),
                    action=np.zeros((n, spec.action_dim), dtype=np.float32),
                )
            ],
            action_length=0.2,
            action_mode=ABS,
        )

    monkeypatch.setattr("lbm.dataloader.custom.make_custom_dataset", _fake_make)
    cfg = TrainConfig()
    cfg.data.data_root_dir = str(tmp_path)
    mix = load_selected_dumps(cfg, ("egoverse", "libero"), mode="train")
    assert seen == ["egoverse"]
    assert len(mix.datasets) == 1


def test_concat_mixture_pads_two_custom_specs():
    import numpy as np

    from lbm.dataloader.custom import CUSTOM_SPECS, CustomMixtureDataset, CustomSingleDataset, Episode
    from lbm.dataloader.pad import collate_fn

    def _ep(spec, seed):
        rng = np.random.default_rng(seed)
        n = 8
        return Episode(
            images={
                cam: rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys
            },
            state=rng.standard_normal((n, spec.state_dim)).astype(np.float32),
            action=rng.standard_normal((n, spec.action_dim)).astype(np.float32),
            lang="pick",
        )

    kai = CustomSingleDataset(
        CUSTOM_SPECS["kai0"], episodes=[_ep(CUSTOM_SPECS["kai0"], 0)], action_length=0.2, action_mode=ABS
    )
    hifi = CustomSingleDataset(
        CUSTOM_SPECS["hifi_umi"], episodes=[_ep(CUSTOM_SPECS["hifi_umi"], 1)], action_length=0.2, action_mode=ABS
    )
    mix = CustomMixtureDataset([(kai, 1.0), (hifi, 1.0)])
    io = mix.policy_io
    assert io["action_dim"] == 20
    assert "top_head" in io["camera_keys"]
    assert "head_main" in io["camera_keys"]
    batch = collate_fn([mix[0], mix[len(kai)]])
    assert batch["action"].shape[0] == 2
    assert batch["action"].shape[-1] == 20
    assert int(batch["embodiment_id"][0]) == 7
    assert int(batch["embodiment_id"][1]) == 16
    assert batch["camera_mask"].shape[1] == len(io["camera_keys"])
