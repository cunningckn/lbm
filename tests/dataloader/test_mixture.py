from tests.fixtures.lerobot_tree import write_lerobot_v2_tree
from tests.fixtures.robot_profiles import robot_io

from lbm.batch import infer_policy_io
from lbm.dataloader.custom import CustomMixtureDataset, make_custom_dataset
from lbm.dataloader.embodiment import embodiment_id_from_tag


def test_synthetic_trees_and_embodiment_ids(tmp_path):
    rm = write_lerobot_v2_tree(tmp_path, robot_type="rmbench", name="rmbench")
    lib = write_lerobot_v2_tree(tmp_path, robot_type="libero", name="libero")
    assert (rm / "meta" / "info.json").is_file()
    assert (lib / "meta" / "info.json").is_file()
    assert robot_io("rmbench").embodiment == "aloha"
    assert robot_io("libero").embodiment == "franka"
    assert embodiment_id_from_tag("aloha") == 7
    assert embodiment_id_from_tag("franka") == 25


def test_mixture_from_synthetic_trees(tmp_path):
    write_lerobot_v2_tree(tmp_path, robot_type="rmbench", name="rmbench")
    write_lerobot_v2_tree(tmp_path, robot_type="libero", name="libero")
    from tests.fixtures.custom_cfg import custom_cfg

    data_cfg = custom_cfg(use_mmap=True, include_state=True)
    rm = make_custom_dataset(tmp_path / "rmbench", "rmbench", data_cfg)
    lib = make_custom_dataset(tmp_path / "libero", "libero", data_cfg)
    mix = CustomMixtureDataset([(rm, 1.0), (lib, 1.0)], mode="train", seed=0)
    assert {ds.spec.embodiment for ds in mix.datasets} == {"aloha", "franka"}
    io = infer_policy_io(mix)
    assert io["action_dim"] == 14
    assert io["state_dim"] == 14
    assert set(io["camera_keys"]) >= {"cam_high", "image"}
    assert embodiment_id_from_tag(rm.spec.embodiment) == 7
    assert embodiment_id_from_tag(lib.spec.embodiment) == 25


def test_catalog_mixture_uses_explicit_data_root(tmp_path, monkeypatch):
    from lbm.config import TrainConfig
    from lbm.dataloader.mixture import load_dataset

    write_lerobot_v2_tree(tmp_path, robot_type="libero", name="libero")
    write_lerobot_v2_tree(tmp_path, robot_type="rmbench", name="rmbench")
    monkeypatch.setenv("LBM_DATASETS", str(tmp_path / "wrong-root"))
    cfg = TrainConfig()
    cfg.data.data_root_dir = str(tmp_path)
    cfg.data.data_mix = "libero,rmbench"
    cfg.data.use_mmap = cfg.data.use_mmap_frames = False
    mix = load_dataset(cfg)
    assert len(mix.datasets) == 2
    assert all(ds.root.is_relative_to(tmp_path) for ds in mix.datasets)
