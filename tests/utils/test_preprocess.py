import importlib.util
from pathlib import Path

import numpy as np
import torch

from lbm.action_space import ABS
from lbm.dataloader.custom import CUSTOM_SPECS, CustomSingleDataset, Episode, make_custom_dataset
from lbm.dataloader.custom.sources import save_numpy_episode
from lbm.utils.preprocess import (
    compute_norm_stats,
    dump_norm_stats_path,
    load_norm_stats,
    norm_stats_filename,
    normalize,
    parse_norm_stats,
    resize_pad_normalize,
    save_norm_stats,
    unnormalize,
)
from tests.fixtures.custom_cfg import custom_cfg

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    path = _SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_roundtrip():
    stats = {"mean": np.array([1.0, 2.0], dtype=np.float32), "std": np.array([0.5, 4.0], dtype=np.float32)}
    x = np.array([[1.0, 2.0], [1.5, 6.0]], dtype=np.float32)
    out = unnormalize(normalize(x, stats), stats)
    np.testing.assert_allclose(out, x, rtol=1e-5, atol=1e-5)


def test_normalize_quantile_maps_q01_q99_to_unit_interval():
    stats = {
        "mean": np.array([0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0], dtype=np.float32),
        "q01": np.array([0.0, 10.0], dtype=np.float32),
        "q99": np.array([2.0, 20.0], dtype=np.float32),
    }
    x = np.array([[0.0, 10.0], [2.0, 20.0], [1.0, 15.0]], dtype=np.float32)
    normed = normalize(x, stats)
    np.testing.assert_allclose(normed[0], [-1.0, -1.0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(normed[1], [1.0, 1.0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(normed[2], [0.0, 0.0], rtol=1e-5, atol=1e-5)
    out = unnormalize(normed, stats)
    np.testing.assert_allclose(out, x, rtol=1e-5, atol=1e-5)
    torch_out = unnormalize(normalize(torch.as_tensor(x), stats), stats)
    np.testing.assert_allclose(torch_out.numpy(), x, rtol=1e-5, atol=1e-5)


def test_parse_norm_stats_unwraps_nested():
    raw = {
        "xdof": {
            "state": {"mean": [0.0], "std": [1.0]},
            "actions": {"mean": [0.5], "std": [2.0]},
        }
    }
    stats = parse_norm_stats(raw)
    np.testing.assert_array_equal(stats["state"]["mean"], np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(stats["actions"]["std"], np.array([2.0], dtype=np.float32))


def test_resize_pad_normalize_to_224():
    img = torch.rand(3, 120, 160)
    out = resize_pad_normalize(img)
    assert out.shape == (3, 224, 224)
    assert torch.isfinite(out).all()


def _tiny_episode(n_frames: int = 8, *, seed: int = 0) -> Episode:
    spec = CUSTOM_SPECS["abc"]
    rng = np.random.default_rng(seed)
    images = {cam: rng.integers(0, 255, size=(n_frames, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys}
    return Episode(
        images=images,
        state=rng.standard_normal((n_frames, spec.state_dim)).astype(np.float32),
        action=rng.standard_normal((n_frames, spec.action_dim)).astype(np.float32),
        lang="test",
    )


def test_compute_norm_stats_roundtrip_in_memory(tmp_path):
    spec = CUSTOM_SPECS["abc"]
    episode = _tiny_episode()
    ds = CustomSingleDataset(spec, [episode], action_length=0.2, action_mode=ABS)
    payload = compute_norm_stats(ds)
    assert payload["spec"] == "abc"
    assert payload["norm_stats"]["actions"]["count"] == episode.action.shape[0]
    path = tmp_path / "norm_stats.json"
    save_norm_stats(path, payload)
    loaded = load_norm_stats(path)
    parsed = parse_norm_stats(payload)
    np.testing.assert_allclose(loaded["state"]["mean"], parsed["state"]["mean"], rtol=1e-5)
    np.testing.assert_allclose(loaded["actions"]["std"], parsed["actions"]["std"], rtol=1e-5)
    np.testing.assert_allclose(
        loaded["state"]["mean"],
        episode.state.mean(axis=0),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        loaded["state"]["q01"],
        np.quantile(episode.state, 0.01, axis=0).astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    from lbm.action_space import apply_action_space_frames, resolve_action_space

    rel_action = apply_action_space_frames(episode.action, episode.state, resolve_action_space(spec, ABS))
    np.testing.assert_allclose(
        loaded["actions"]["q99"],
        np.quantile(rel_action, 0.99, axis=0).astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "action" not in payload["norm_stats"]
    assert "actions" in payload["norm_stats"]


def test_compute_norm_script_writes_json(tmp_path):
    folder = tmp_path / "abc"
    folder.mkdir()
    save_numpy_episode(folder / "episode_000.npz", _tiny_episode())
    compute_norm = _load_script("compute_norm.py")
    compute_norm.main(["--dataset", "abc", "--data-root", str(tmp_path)])
    from lbm.action_space import resolve_action_space
    from lbm.dataloader.custom.datasets import CUSTOM_SPECS

    spec = CUSTOM_SPECS["abc"]
    cfg = custom_cfg(action_mode="delta", action_length=5.0)
    ds = make_custom_dataset(folder, "abc", data_cfg=cfg)
    path = dump_norm_stats_path(folder, ds.action_freq, ds.action_length, resolve_action_space(spec, ds.action_mode))
    assert path.name == "norm_stats_joint_delta_10hz_5s.json"
    assert path.is_file()
    stats = load_norm_stats(path)
    assert stats["state"]["mean"].shape == (CUSTOM_SPECS["abc"].state_dim,)
    assert stats["actions"]["mean"].shape == (CUSTOM_SPECS["abc"].action_dim,)
    assert stats["state"]["q01"].shape == stats["state"]["mean"].shape
    assert stats["actions"]["q99"].shape == stats["actions"]["mean"].shape
    reloaded = make_custom_dataset(folder, "abc", data_cfg=cfg)
    assert reloaded.norm_stats is not None
    np.testing.assert_allclose(reloaded.norm_stats["state"]["q01"], stats["state"]["q01"])


def test_prebuild_mmap_script_noops_on_numpy_dump(tmp_path):
    folder = tmp_path / "abc"
    folder.mkdir()
    save_numpy_episode(folder / "episode_000.npz", _tiny_episode())
    prebuild = _load_script("prebuild_mmap.py")
    prebuild.main(["--dataset", "abc", "--data-root", str(tmp_path), "--workers", "0"])
    assert not list(folder.rglob("frames.bin"))


def test_compute_norm_mmap_flag_noops_on_numpy_dump(tmp_path):
    folder = tmp_path / "abc"
    folder.mkdir()
    save_numpy_episode(folder / "episode_000.npz", _tiny_episode())
    compute_norm = _load_script("compute_norm.py")
    compute_norm.main(["--dataset", "abc", "--data-root", str(tmp_path), "--mmap", "--workers", "0"])
    ds = make_custom_dataset(folder, "abc", data_cfg=custom_cfg(action_mode="delta", action_length=5.0))
    from lbm.action_space import resolve_action_space

    path = dump_norm_stats_path(
        folder, ds.action_freq, ds.action_length, resolve_action_space(ds.spec, ds.action_mode)
    )
    assert path.is_file()
    assert not list(folder.rglob("frames.bin"))


def test_compute_norm_reads_lerobot_parquet_mmap(tmp_path):
    from tests.fixtures.lerobot_tree import write_lerobot_v2_tree

    root = write_lerobot_v2_tree(tmp_path, robot_type="kai0", name="kai0")
    ds_mmap = make_custom_dataset(root, "kai0", data_cfg=custom_cfg(use_mmap=True))
    ds_raw = make_custom_dataset(root, "kai0", data_cfg=custom_cfg())
    mmap_state, mmap_action = ds_mmap._vectors(0)
    raw_state, raw_action = ds_raw._vectors(0)
    np.testing.assert_allclose(mmap_state, raw_state, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mmap_action, raw_action, rtol=1e-5, atol=1e-5)
    assert list(root.rglob("*.npy")), "expected parquet mmap .npy under .mmap/"

    compute_norm = _load_script("compute_norm.py")
    compute_norm.main(["--dataset", "kai0", "--data-root", str(tmp_path), "--mmap", "--workers", "0"])
    ds = make_custom_dataset(root, "kai0", data_cfg=custom_cfg(use_mmap=True, action_mode="delta", action_length=5.0))
    from lbm.action_space import resolve_action_space

    stats = load_norm_stats(
        dump_norm_stats_path(root, ds.action_freq, ds.action_length, resolve_action_space(ds.spec, ds.action_mode))
    )
    np.testing.assert_allclose(stats["state"]["mean"], raw_state.mean(axis=0), rtol=1e-5, atol=1e-5)


def test_norm_stats_filename_adds_length_only_for_computed_delta():
    from lbm.action_space import bimanual_joint, delta_eef, dual_eef

    assert norm_stats_filename(10.0, 5.0, slices=delta_eef()) == "norm_stats_eef_delta_10hz.json"
    assert norm_stats_filename(10.0, 5.0, slices=dual_eef()) == "norm_stats_eef_delta_10hz_5s.json"
    assert norm_stats_filename(10.0, 5.0, slices=bimanual_joint()) == "norm_stats_joint_delta_10hz_5s.json"
    assert norm_stats_filename(10.0, 5.0, slices=bimanual_joint(rep="rel")) == "norm_stats_joint_rel_10hz.json"


def test_progress_disabled_under_pytest():
    from lbm.utils.progress import progress_enabled, track

    assert progress_enabled() is False
    assert list(track(range(3), desc="x")) == [0, 1, 2]
