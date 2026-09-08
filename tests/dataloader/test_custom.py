"""Custom dump specs and loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lbm.action_space import ABS
from lbm.batch import infer_policy_io, merge_policy_io
from lbm.dataloader.custom import (
    CUSTOM_MIXTURES,
    CUSTOM_SPECS,
    CustomMixtureDataset,
    CustomSingleDataset,
    Episode,
    load_episodes,
    make_custom_dataset,
    uses_custom_backend,
)
from lbm.dataloader.custom.spec import CustomSpec
from lbm.dataloader.pad import collate_fn
from tests.fixtures.custom_cfg import custom_cfg
from tests.fixtures.lerobot_tree import lerobot_info

_EXPECTED = {
    "agibot": ("agibot_genie1", ("top_head", "hand_left", "hand_right"), 20, 22, 30.0, 26),
    "galaxea": ("galaxea", ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"), 32, 14, 15.0, 11),
    "kai0": ("aloha", ("top_head", "hand_left", "hand_right"), 14, 14, 30.0, 7),
    "egoverse": ("egoverse", ("cam_high", "cam_left_wrist", "cam_right_wrist"), 16, 16, 30.0, 12),
    "das_gripper": ("das_gripper", ("cam_high", "cam_left_wrist", "cam_right_wrist"), 16, 16, 30.0, 13),
    "droid": ("oxe_droid", ("exterior_1_left", "exterior_2_left", "wrist_left"), 8, 8, 15.0, 17),
    "hy_lance": ("hy_lance", ("cam_high", "cam_left_wrist", "cam_right_wrist"), 16, 16, 30.0, 14),
    "hifi_umi": ("hifi_umi", ("head_main", "left_hand_up", "right_hand_up"), 20, 20, 25.0, 16),
    "abc": ("abc", ("cam_high", "cam_left_wrist", "cam_right_wrist"), 14, 14, 10.0, 20),
    "libero": ("franka", ("image", "wrist_image"), 8, 7, 10.0, 25),
    "rmbench": ("aloha", ("cam_high", "cam_left_wrist", "cam_right_wrist"), 14, 14, 10.0, 7),
    "robotwin": ("aloha", ("cam_high", "cam_left_wrist", "cam_right_wrist"), 14, 14, 30.0, 7),
}


def _episode(spec: CustomSpec, n_frames: int = 12, *, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed)
    images = {
        cam: rng.integers(0, 255, size=(n_frames, 8, 8, 3), dtype=np.uint8)
        for cam in spec.camera_keys
    }
    return Episode(
        images=images,
        state=rng.standard_normal((n_frames, spec.state_dim)).astype(np.float32),
        action=rng.standard_normal((n_frames, spec.action_dim)).astype(np.float32),
        lang="pick up the cup",
    )


def test_pad_uses_independent_embodiment_table():
    pad = Path("src/lbm/dataloader/pad.py").read_text(encoding="utf-8")
    assert "lbm.dataloader.embodiment" in pad
    from lbm.dataloader.embodiment import EMBODIMENT_IDS, embodiment_id_from_tag

    assert embodiment_id_from_tag("aloha") == 7
    assert embodiment_id_from_tag("franka") == 25
    assert all(int(v) < 32 for v in EMBODIMENT_IDS.values())
    for spec in CUSTOM_SPECS.values():
        assert spec.embodiment_id == embodiment_id_from_tag(spec.embodiment)


def test_custom_specs_are_registered():
    for name, (embodiment, cameras, state_dim, action_dim, fps, eid) in _EXPECTED.items():
        spec = CUSTOM_SPECS[name]
        assert spec.name == name
        assert spec.embodiment == embodiment
        assert spec.camera_keys == cameras
        assert spec.state_dim == state_dim
        assert spec.action_dim == action_dim
        assert spec.fps == fps
        assert spec.embodiment_id == eid
        assert name in CUSTOM_MIXTURES
    assert "libero_all" not in CUSTOM_MIXTURES
    assert CUSTOM_MIXTURES["robotwin"] == [("robotwin", 1.0, "robotwin")]
    from lbm.dataloader.custom.datasets.mixes import ALL, NAMED_MIXES

    assert [folder for folder, _w, spec in CUSTOM_MIXTURES["all"]] == list(ALL)
    assert CUSTOM_MIXTURES["all"] == list(NAMED_MIXES["all"])
    assert set(ALL) <= set(CUSTOM_SPECS)
    assert "robodojo" not in CUSTOM_SPECS


def test_single_dataset_applies_quantile_norm():
    spec = CUSTOM_SPECS["kai0"]
    episode = _episode(spec)
    stats = {
        "state": {
            "q01": np.zeros(spec.state_dim, dtype=np.float32),
            "q99": np.full(spec.state_dim, 2.0, dtype=np.float32),
        },
        "actions": {
            "q01": np.zeros(spec.action_dim, dtype=np.float32),
            "q99": np.full(spec.action_dim, 2.0, dtype=np.float32),
        },
    }
    raw = CustomSingleDataset(spec, [episode], action_length=0.2, action_mode=ABS)
    normed = CustomSingleDataset(spec, [episode], action_length=0.2, norm_stats=stats, action_mode=ABS)
    from lbm.utils.preprocess import normalize

    np.testing.assert_allclose(normed[0]["state"], normalize(raw[0]["state"], stats["state"]), rtol=1e-5)
    np.testing.assert_allclose(normed[0]["action"], normalize(raw[0]["action"], stats["actions"]), rtol=1e-5)


def test_single_dataset_packed_sample():
    spec = CUSTOM_SPECS["kai0"]
    ds = CustomSingleDataset(spec, [_episode(spec)], action_length=0.2, action_mode=ABS)
    sample = ds[0]
    assert sample["action"].shape[-1] == 14
    assert sample["state"].shape[-1] == 14
    assert sample["camera_keys"] == spec.camera_keys
    assert len(sample["image"]) == 3
    assert sample["lang"] == "pick up the cup"
    assert int(sample["embodiment_id"]) == 7
    assert sample["robot_tag"] == "aloha"
    io = infer_policy_io(ds)
    assert io["action_dim"] == 14
    assert io["state_dim"] == 14
    assert io["camera_keys"] == spec.camera_keys
    assert io["chunk_length"] == ds.chunk_length


def test_mixture_pads_heterogeneous_custom_specs():
    kai0 = CustomSingleDataset(CUSTOM_SPECS["kai0"], [_episode(CUSTOM_SPECS["kai0"], seed=1)], action_length=0.2, action_mode=ABS)
    agi = CustomSingleDataset(CUSTOM_SPECS["agibot"], [_episode(CUSTOM_SPECS["agibot"], seed=2)], action_length=0.2, action_mode=ABS)
    mix = CustomMixtureDataset([(kai0, 1.0), (agi, 1.0)])
    io = infer_policy_io(mix)
    assert io["action_dim"] == 22
    assert io["state_dim"] == 20
    assert set(io["camera_keys"]) >= {"top_head", "hand_left", "hand_right"}
    batch = collate_fn([kai0[0], agi[0]])
    assert batch["action"].shape[-1] == 22
    assert batch["state"].shape[-1] == 20
    assert int(batch["embodiment_id"][0]) == 7
    assert int(batch["embodiment_id"][1]) == 26
    assert batch["action_mask"].shape == batch["action"].shape


def test_numpy_episode_roundtrip(tmp_path):
    spec = CUSTOM_SPECS["abc"]
    episode = _episode(spec, n_frames=8)
    folder = tmp_path / "abc"
    folder.mkdir()
    from lbm.dataloader.custom.sources import save_numpy_episode

    save_numpy_episode(folder / "episode_000.npz", episode)
    loaded = load_episodes(folder, spec)
    assert len(loaded) == 1
    assert loaded[0].state.shape == (8, 14)
    assert loaded[0].action.shape == (8, 14)
    ds = make_custom_dataset(folder, "abc", data_cfg=custom_cfg(action_length=0.5))
    assert len(ds) >= 1
    assert ds[0]["action"].shape[-1] == 14


def test_lerobot_tree_without_modality_json(tmp_path):
    spec = CUSTOM_SPECS["kai0"]
    root = tmp_path / "kai0"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 10
    (meta / "info.json").write_text(
        json.dumps(lerobot_info(version="v2.1", fps=30, robot_type="agilex", total_episodes=1))
    )
    (meta / "episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["pick"],"length":10}\n'
    )
    (meta / "tasks.jsonl").write_text('{"task_index":0,"task":"pick"}\n')
    rows = {
        "observation.state": [np.ones(14, dtype=np.float32) * i for i in range(n)],
        "action": [np.zeros(14, dtype=np.float32) + i for i in range(n)],
        "task_index": np.zeros(n, dtype=np.int64),
        "timestamp": np.arange(n, dtype=np.float64) / 30.0,
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.zeros(n, dtype=np.int64),
    }
    pd.DataFrame(rows).to_parquet(data / "episode_000000.parquet")
    from tests.fixtures.lerobot_tree import _write_dummy_mp4

    for cam in spec.camera_keys:
        _write_dummy_mp4(root / "videos" / "chunk-000" / f"observation.images.{cam}" / "episode_000000.mp4", n)
    ds = make_custom_dataset(root, "kai0", data_cfg=custom_cfg())
    sample = ds[0]
    assert sample["state"].shape[-1] == 14
    assert sample["action"].shape[-1] == 14
    assert len(sample["image"]) == len(spec.camera_keys)


def test_scan_lerobot_requires_codebase_version(tmp_path):
    from lbm.dataloader.custom.common.lerobot import scan_lerobot

    root = tmp_path / "kai0"
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = lerobot_info(version="v2.1", fps=30)
    del info["codebase_version"]
    (meta / "info.json").write_text(json.dumps(info))
    with pytest.raises(KeyError, match="codebase_version"):
        scan_lerobot(root, CUSTOM_SPECS["kai0"])


def test_scan_lerobot_rejects_unknown_version(tmp_path):
    from lbm.dataloader.custom.common.lerobot import scan_lerobot

    root = tmp_path / "kai0"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(lerobot_info(version="v4.0", fps=30)))
    with pytest.raises(ValueError, match="unsupported codebase_version"):
        scan_lerobot(root, CUSTOM_SPECS["kai0"])


def test_lerobot_missing_spec_key_raises(tmp_path):
    from tests.fixtures.lerobot_tree import write_lerobot_v2_tree

    root = write_lerobot_v2_tree(tmp_path, robot_type="rmbench", name="rm")
    ds = make_custom_dataset(root, "libero", data_cfg=custom_cfg())
    with pytest.raises(KeyError, match="state"):
        ds._vectors(0)


def test_libero_parquet_png_columns(tmp_path):
    cv2 = pytest.importorskip("cv2")
    spec = CUSTOM_SPECS["libero"]
    root = tmp_path / "libero"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 4
    (meta / "info.json").write_text(json.dumps(lerobot_info(version="v2.0", fps=10, robot_type="panda")))
    (meta / "episodes.jsonl").write_text('{"episode_index":0,"tasks":["pick"],"length":4}\n')
    ok, buf = cv2.imencode(".png", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    cell = {"bytes": buf.tobytes()}
    rows = {
        "state": [np.ones(8, dtype=np.float32) * i for i in range(n)],
        "actions": [np.zeros(7, dtype=np.float32) + i for i in range(n)],
        "image": [cell] * n,
        "wrist_image": [cell] * n,
        "task_index": np.zeros(n, dtype=np.int64),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.zeros(n, dtype=np.int64),
    }
    pd.DataFrame(rows).to_parquet(data / "episode_000000.parquet")
    ds = make_custom_dataset(root, "libero", data_cfg=custom_cfg())
    sample = ds[0]
    assert sample["state"].shape[-1] == 8
    assert sample["action"].shape[-1] == 7
    assert len(sample["image"]) == 2
    assert sample["camera_keys"] == spec.camera_keys


def test_libero_prebuild_writes_parquet_and_jpeg_mmap(tmp_path):
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "libero"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 4
    (meta / "info.json").write_text(json.dumps(lerobot_info(version="v2.0", fps=10, robot_type="panda")))
    (meta / "episodes.jsonl").write_text('{"episode_index":0,"tasks":["pick"],"length":4}\n')
    ok, buf = cv2.imencode(".png", np.full((16, 16, 3), 80, dtype=np.uint8))
    assert ok
    cell = {"bytes": buf.tobytes()}
    rows = {
        "state": [np.ones(8, dtype=np.float32) * i for i in range(n)],
        "actions": [np.zeros(7, dtype=np.float32) + i for i in range(n)],
        "image": [cell] * n,
        "wrist_image": [cell] * n,
        "task_index": np.zeros(n, dtype=np.int64),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.zeros(n, dtype=np.int64),
    }
    parquet = data / "episode_000000.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    ds = make_custom_dataset(
        root,
        "libero",
        data_cfg=custom_cfg(use_mmap=True, use_mmap_frames=True),
    )
    ds.prebuild_mmap_caches(workers=0)
    parquet_cache = root / ".mmap" / "data__chunk-000__episode_000000"
    assert (parquet_cache / "manifest.json").is_file()
    assert (parquet_cache / "state.npy").is_file()
    assert not (parquet_cache / "image.npy").exists()
    frames_root = root / ".mmap" / "frames" / "episode_000000"
    assert (frames_root / "image" / "frames.bin").is_file()
    assert (frames_root / "wrist_image" / "frames.bin").is_file()
    sample = ds[0]
    assert sample["image"][0].shape[-1] == 3


def test_libero_falls_back_to_parquet_images_when_frame_cache_missing(tmp_path):
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "libero"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 4
    (meta / "info.json").write_text(json.dumps(lerobot_info(version="v2.0", fps=10, robot_type="panda")))
    (meta / "episodes.jsonl").write_text('{"episode_index":0,"tasks":["pick"],"length":4}\n')
    ok, buf = cv2.imencode(".png", np.full((16, 16, 3), 90, dtype=np.uint8))
    assert ok
    cell = {"bytes": buf.tobytes()}
    pd.DataFrame(
        {
            "state": [np.ones(8, dtype=np.float32) * i for i in range(n)],
            "actions": [np.zeros(7, dtype=np.float32) + i for i in range(n)],
            "image": [cell] * n,
            "wrist_image": [cell] * n,
            "task_index": np.zeros(n, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.zeros(n, dtype=np.int64),
        }
    ).to_parquet(data / "episode_000000.parquet")
    ds = make_custom_dataset(
        root,
        "libero",
        data_cfg=custom_cfg(use_mmap=True, use_mmap_frames=True),
    )
    ds.set_mmap_allow_build(False)
    sample = ds[0]
    assert sample["image"][0].shape[-1] == 3
    assert int(sample["image"][0].max()) > 0
    assert not list(root.rglob("frames.bin"))


def test_libero_missing_camera_raises(tmp_path):
    root = tmp_path / "libero"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 2
    (meta / "info.json").write_text(json.dumps(lerobot_info(version="v2.0", fps=10, robot_type="panda")))
    (meta / "episodes.jsonl").write_text('{"episode_index":0,"tasks":["pick"],"length":2}\n')
    pd.DataFrame(
        {
            "state": [np.ones(8, dtype=np.float32) for _ in range(n)],
            "actions": [np.zeros(7, dtype=np.float32) for _ in range(n)],
            "task_index": np.zeros(n, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.zeros(n, dtype=np.int64),
        }
    ).to_parquet(data / "episode_000000.parquet")
    ds = make_custom_dataset(
        root,
        "libero",
        data_cfg=custom_cfg(),
    )
    with pytest.raises(KeyError, match="camera"):
        ds[0]


def test_lerobot_v2_parquet_images_without_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "libero"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 3
    features = {
        "observation.images.image": {"dtype": "image", "shape": [16, 16, 3]},
        "observation.images.wrist_image": {"dtype": "image", "shape": [16, 16, 3]},
    }
    (meta / "info.json").write_text(
        json.dumps(lerobot_info(version="v2.0", fps=10, robot_type="panda", features=features))
    )
    (meta / "episodes.jsonl").write_text('{"episode_index":0,"tasks":["pick"],"length":3}\n')
    ok, buf = cv2.imencode(".png", np.full((16, 16, 3), 60, dtype=np.uint8))
    assert ok
    cell = {"bytes": buf.tobytes()}
    pd.DataFrame(
        {
            "state": [np.ones(8, dtype=np.float32) for _ in range(n)],
            "actions": [np.zeros(7, dtype=np.float32) for _ in range(n)],
            "image": [cell] * n,
            "wrist_image": [cell] * n,
            "task_index": np.zeros(n, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.zeros(n, dtype=np.int64),
        }
    ).to_parquet(data / "episode_000000.parquet")
    ds = make_custom_dataset(
        root,
        "libero",
        data_cfg=custom_cfg(),
    )
    rec = ds.records[0]
    assert rec.kind == "lerobot_v2"
    from lbm.dataloader.custom.common.lerobot import lerobot_of

    assert lerobot_of(rec) is not None and not lerobot_of(rec).videos
    sample = ds[0]
    assert int(sample["image"][0].max()) > 0


def test_lerobot_v3_parquet_images_without_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "libero_v3"
    meta_ep = root / "meta" / "episodes" / "chunk-000"
    data = root / "data" / "chunk-000"
    meta_ep.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 3
    features = {
        "observation.images.image": {"dtype": "image", "shape": [16, 16, 3]},
        "observation.images.wrist_image": {"dtype": "image", "shape": [16, 16, 3]},
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(lerobot_info(version="v3.0", fps=10, robot_type="panda", features=features))
    )
    pd.DataFrame(
        [{"episode_index": 0, "length": n, "tasks": [["pick"]], "data/chunk_index": 0, "data/file_index": 0}]
    ).to_parquet(meta_ep / "file-000.parquet")
    ok, buf = cv2.imencode(".png", np.full((16, 16, 3), 70, dtype=np.uint8))
    assert ok
    cell = {"bytes": buf.tobytes()}
    pd.DataFrame(
        {
            "state": [np.ones(8, dtype=np.float32) for _ in range(n)],
            "actions": [np.zeros(7, dtype=np.float32) for _ in range(n)],
            "image": [cell] * n,
            "wrist_image": [cell] * n,
            "episode_index": np.zeros(n, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
        }
    ).to_parquet(data / "file-000.parquet")
    ds = make_custom_dataset(
        root,
        "libero",
        data_cfg=custom_cfg(),
    )
    rec = ds.records[0]
    assert rec.kind == "lerobot_v3"
    from lbm.dataloader.custom.common.lerobot import lerobot_of

    assert lerobot_of(rec) is not None and not lerobot_of(rec).videos
    sample = ds[0]
    assert sample["image"][0].shape[-1] == 3
    assert int(sample["image"][0].max()) > 0


def test_uses_custom_backend_routing():
    assert uses_custom_backend(robot_type="agibot")
    assert uses_custom_backend(data_mix="galaxea")
    assert uses_custom_backend(dataset="kai0")
    assert uses_custom_backend(dataset="/data/dumps/kai0")
    assert uses_custom_backend(robot_type="rmbench")
    assert uses_custom_backend(data_mix="libero")
    assert uses_custom_backend(data_mix="robotwin")
    assert uses_custom_backend(dataset="robotwin")
    assert uses_custom_backend(dataset="libero")
    assert uses_custom_backend(data_mix="all")
    assert uses_custom_backend(data_mix="custom_all")
    assert uses_custom_backend(data_mix="kai0,galaxea")
    assert uses_custom_backend(data_mix="kai0,libero")
    assert not uses_custom_backend(data_mix="libero_all")
    assert not uses_custom_backend(robot_type="not_a_robot")


def test_merge_policy_io_from_custom_specs():
    ios = [CUSTOM_SPECS[name].as_policy_io(chunk_length=8) for name in ("kai0", "hifi_umi")]
    merged = merge_policy_io(ios)
    assert merged["action_dim"] == 20
    assert merged["state_dim"] == 20
    assert "top_head" in merged["camera_keys"]
    assert "head_main" in merged["camera_keys"]


def test_unknown_spec_raises():
    with pytest.raises(KeyError):
        make_custom_dataset("/tmp/none", "not_a_robot", data_cfg=custom_cfg())


def test_lerobot_v2_chunk_from_episode_index(tmp_path):
    from lbm.dataloader.custom.common.lerobot import lerobot_of, resolve_lerobot_video, scan_lerobot
    from tests.fixtures.lerobot_tree import _write_dummy_mp4

    spec = CUSTOM_SPECS["kai0"]
    root = tmp_path / "kai0"
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            f"observation.images.{cam}": {"dtype": "video", "shape": [3, 16, 16]} for cam in spec.camera_keys
        },
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["first"],"length":2}\n'
        '{"episode_index":1000,"tasks":["second"],"length":2}\n'
    )
    for epi, chunk in ((0, 0), (1000, 1)):
        data = root / "data" / f"chunk-{chunk:03d}"
        data.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "observation.state": [np.ones(14, np.float32)] * 2,
                "action": [np.zeros(14, np.float32)] * 2,
                "episode_index": np.full(2, epi, np.int64),
            }
        ).to_parquet(data / f"episode_{epi:06d}.parquet")
        for cam in spec.camera_keys:
            _write_dummy_mp4(
                root / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{cam}" / f"episode_{epi:06d}.mp4",
                2,
            )
    recs = scan_lerobot(root, spec)
    dumps = [lerobot_of(r) for r in recs]
    assert [d.episode_index for d in dumps] == [0, 1000]
    assert dumps[1].chunk == 1
    assert recs[1].path.endswith("chunk-001/episode_001000.parquet")
    found = resolve_lerobot_video(recs[1], spec, "top_head")
    assert found is not None
    assert "chunk-001" in found[0].as_posix()


def test_lerobot_v3_packed_video_from_timestamp(tmp_path):
    from lbm.dataloader.custom.common.lerobot import lerobot_of
    from lbm.dataloader.custom.mmap_frames import decode_kwargs, resolve_mmap_frames
    from tests.fixtures.lerobot_tree import _write_dummy_mp4

    spec = CUSTOM_SPECS["hifi_umi"]
    root = tmp_path / "hifi"
    meta_ep = root / "meta" / "episodes" / "chunk-000"
    data = root / "data" / "chunk-000"
    meta_ep.mkdir(parents=True)
    data.mkdir(parents=True)
    fps = 25.0
    n0, n1 = 4, 4
    cams = spec.camera_keys
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": fps,
                "features": {f"observation.images.{cam}": {"dtype": "video", "shape": [3, 16, 16]} for cam in cams},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            }
        )
    )
    rows = []
    for epi, n, origin in ((0, n0, 0.0), (1, n1, n0 / fps)):
        row = {
            "episode_index": epi,
            "length": n,
            "tasks": [["fold"]],
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for cam in cams:
            key = f"observation.images.{cam}"
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = 0
            row[f"videos/{key}/from_timestamp"] = origin
        rows.append(row)
    pd.DataFrame(rows).to_parquet(meta_ep / "file-000.parquet")
    n = n0 + n1
    pd.DataFrame(
        {
            "observation.state": [np.ones(20, np.float32) for _ in range(n)],
            "action": [np.zeros(20, np.float32) for _ in range(n)],
            "episode_index": np.array([0] * n0 + [1] * n1, dtype=np.int64),
            "frame_index": np.arange(n),
        }
    ).to_parquet(data / "file-000.parquet")
    for cam in cams:
        _write_dummy_mp4(root / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4", n)
    ds = make_custom_dataset(
        root,
        "hifi_umi",
        data_cfg=custom_cfg(action_length=0.04, use_mmap=True, use_mmap_frames=True),
    )
    assert len(ds.records) == 2
    assert lerobot_of(ds.records[1]).videos["head_main"].from_timestamp == pytest.approx(n0 / fps)
    ref = resolve_mmap_frames(ds.records[1], spec, "head_main")
    assert ref is not None
    assert decode_kwargs(ref, spec)["mode"] == "lerobot"
    live0 = int(ds[0]["image"][0].max())
    live1 = int(ds[n0]["image"][0].max())
    assert live1 != live0
    ds.prebuild_mmap_caches(workers=0)
    assert int(ds[n0]["image"][0].max()) == live1


def test_lerobot_v2_dotted_image_column(tmp_path):
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "libero"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 2
    features = {
        "observation.images.image": {"dtype": "image", "shape": [16, 16, 3]},
        "observation.images.wrist_image": {"dtype": "image", "shape": [16, 16, 3]},
    }
    (meta / "info.json").write_text(
        json.dumps(lerobot_info(version="v2.0", fps=10, robot_type="panda", features=features))
    )
    (meta / "episodes.jsonl").write_text('{"episode_index":0,"tasks":["pick"],"length":2}\n')
    ok, buf = cv2.imencode(".png", np.full((16, 16, 3), 90, dtype=np.uint8))
    assert ok
    cell = {"bytes": buf.tobytes()}
    pd.DataFrame(
        {
            "state": [np.ones(8, np.float32)] * n,
            "actions": [np.zeros(7, np.float32)] * n,
            "observation.images.image": [cell] * n,
            "observation.images.wrist_image": [cell] * n,
            "episode_index": np.zeros(n, np.int64),
        }
    ).to_parquet(data / "episode_000000.parquet")
    ds = make_custom_dataset(
        root,
        "libero",
        data_cfg=custom_cfg(),
    )
    assert int(ds[0]["image"][0].max()) > 0
    from lbm.dataloader.custom.mmap_frames import decode_kwargs, resolve_mmap_frames
    from lbm.dataloader.custom.scan import source_jpegs_for_mmap

    spec = CUSTOM_SPECS["libero"]
    ref = resolve_mmap_frames(ds.records[0], spec, "image")
    assert ref is not None
    blobs = source_jpegs_for_mmap(decode_kwargs(ref, spec))
    assert blobs is not None and len(blobs) == n
    assert blobs[0][:8] == b"\x89PNG\r\n\x1a\n"
