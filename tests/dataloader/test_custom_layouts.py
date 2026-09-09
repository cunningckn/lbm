"""Synthetic dumps that mirror the 10 on-disk layouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lbm.dataloader.custom import CUSTOM_SPECS, make_custom_dataset
from lbm.dataloader.custom.scan import scan_root
from tests.fixtures.custom_cfg import custom_cfg
from tests.fixtures.lerobot_tree import lerobot_info


def _mp4(path: Path, n: int = 4, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv required")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for i in range(n):
        writer.write(np.full((size, size, 3), (i * 40) % 255, dtype=np.uint8))
    writer.release()


def _v2_repo(root: Path, *, cams: tuple[str, ...], state_col: str, action_col: str, n: int = 6) -> Path:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    info = lerobot_info(version="v2.1", fps=30, features={})
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["pick"], "length": n}) + "\n")
    rows = {
        state_col: [np.ones(CUSTOM_SPECS["kai0"].state_dim, np.float32) * i for i in range(n)],
        action_col: [np.zeros(CUSTOM_SPECS["kai0"].action_dim, np.float32) + i for i in range(n)],
        "task_index": np.zeros(n, np.int64),
        "frame_index": np.arange(n),
        "episode_index": np.zeros(n, np.int64),
    }
    pd.DataFrame(rows).to_parquet(data / "episode_000000.parquet")
    for cam in cams:
        _mp4(root / "videos" / "chunk-000" / f"observation.images.{cam}" / "episode_000000.mp4", n)
        _mp4(root / "videos" / "chunk-000" / cam / "episode_000000.mp4", n)
    return root


def test_nested_kai0_layout(tmp_path):
    repo = _v2_repo(
        tmp_path / "Task_A" / "advantage",
        cams=("top_head", "hand_left", "hand_right"),
        state_col="observation.state",
        action_col="action",
    )
    ds = make_custom_dataset(tmp_path, "kai0", data_cfg=custom_cfg(max_episodes=2))
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["action"].shape[-1] == 14
    assert len(sample["image"]) == 3
    recs = scan_root(tmp_path, CUSTOM_SPECS["kai0"])
    assert recs[0].path.endswith("episode_000000.parquet")
    assert repo.is_dir()


def test_custom_lerobot_writes_shared_mmap(tmp_path):
    _v2_repo(
        tmp_path / "Task_A" / "advantage",
        cams=("top_head", "hand_left", "hand_right"),
        state_col="observation.state",
        action_col="action",
        n=4,
    )
    ds = make_custom_dataset(
        tmp_path, "kai0", data_cfg=custom_cfg(use_mmap_frames=True, max_episodes=1)
    )
    sample = ds[0]
    assert sample["image"][0].shape[-1] == 3
    caches = list(tmp_path.rglob("frames.bin"))
    assert caches, "expected JPEG mmap cache under the LeRobot repo"


def test_agibot_hdf5_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5 = tmp_path / "proprio_stats" / "327" / "1" / "proprio_stats.h5"
    h5.parent.mkdir(parents=True)
    t = 5
    with h5py.File(h5, "w") as f:
        f.create_dataset("state/joint/position", data=np.zeros((t, 14), np.float32))
        f.create_dataset("state/effector/position", data=np.zeros((t, 2), np.float32))
        f.create_dataset("state/head/position", data=np.zeros((t, 2), np.float32))
        f.create_dataset("state/waist/position", data=np.zeros((t, 2), np.float32))
        f.create_dataset("action/joint/position", data=np.ones((t, 14), np.float32))
        f.create_dataset("action/effector/position", data=np.ones((t, 2), np.float32))
        f.create_dataset("action/head/position", data=np.ones((t, 2), np.float32))
        f.create_dataset("action/waist/position", data=np.ones((t, 2), np.float32))
        f.create_dataset("action/robot/velocity", data=np.ones((t, 2), np.float32))
    videos = tmp_path / "observations" / "327" / "1" / "videos"
    _mp4(videos / "head_color.mp4", t)
    _mp4(videos / "hand_left_color.mp4", t)
    _mp4(videos / "hand_right_color.mp4", t)
    ds = make_custom_dataset(tmp_path, "agibot", data_cfg=custom_cfg(use_mmap=True, use_mmap_frames=True))
    sample = ds[0]
    assert sample["state"].shape[-1] == 20
    assert sample["action"].shape[-1] == 22
    assert list((tmp_path / ".mmap").rglob("frames.bin")), "agibot mp4 should land in dump-root JPEG mmap"
    assert not list(videos.rglob("frames.bin"))


def test_agibot_scan_skips_dot_cache(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    from lbm.dataloader.custom.datasets.agibot import scan

    def _touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("state/joint/position", data=np.zeros((3, 14), np.float32))

    real = tmp_path / "AgiBotWorld_beta" / "proprio_stats" / "327" / "1" / "proprio_stats.h5"
    cached = tmp_path / ".cache" / "proprio_stats" / "327" / "9" / "proprio_stats.h5"
    nested = tmp_path / "AgiBotWorld_beta" / "proprio_stats" / ".cache" / "327" / "8" / "x.h5"
    _touch(real)
    _touch(cached)
    _touch(nested)
    recs = scan(tmp_path, CUSTOM_SPECS["agibot"])
    assert [rec.path for rec in recs] == [str(real)]


def test_das_hdf5_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    ep = tmp_path / "task" / "00001" / "01706"
    ep.mkdir(parents=True)
    t = 4
    with h5py.File(ep / "episode.hdf5", "w") as f:
        f.create_dataset("observations/left_eef_pose", data=np.zeros((t, 7), np.float64))
        f.create_dataset("observations/right_eef_pose", data=np.zeros((t, 7), np.float64))
        f.create_dataset("observations/mag_left", data=np.zeros(t, np.float64))
        f.create_dataset("observations/mag_right", data=np.ones(t, np.float64))
    _mp4(ep / "cam_left_wrist.mp4", t)
    _mp4(ep / "cam_right_wrist.mp4", t)
    ds = make_custom_dataset(tmp_path, "das_gripper", data_cfg=custom_cfg(use_mmap=True, use_mmap_frames=True))
    sample = ds[0]
    assert sample["state"].shape[-1] == 16
    assert sample["action"].shape[-1] == 16
    assert sample["camera_mask"].tolist() == [False, True, True]
    assert int(np.asarray(sample["image"][0]).max()) == 0
    assert list((tmp_path / ".mmap").rglob("frames.bin")), "das wrist mp4 should land in dump-root JPEG mmap"
    assert not list(ep.rglob("frames.bin"))


def _das_episode(folder: Path, t: int = 4) -> Path:
    h5py = pytest.importorskip("h5py")
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(folder / "episode.hdf5", "w") as handle:
        handle.create_dataset("observations/left_eef_pose", data=np.zeros((t, 7), np.float64))
        handle.create_dataset("observations/right_eef_pose", data=np.zeros((t, 7), np.float64))
        handle.create_dataset("observations/mag_left", data=np.zeros(t, np.float64))
        handle.create_dataset("observations/mag_right", data=np.ones(t, np.float64))
    _mp4(folder / "cam_left_wrist.mp4", t)
    _mp4(folder / "cam_right_wrist.mp4", t)
    return folder / "episode.hdf5"


def test_das_scan_uses_task_metas_not_dir_depth(tmp_path: Path):
    from lbm.dataloader.custom.datasets.das_gripper import scan

    clutter = _das_episode(tmp_path / "Clutter Tidy-Up [Stage2]" / "00001" / "01706")
    cook = _das_episode(tmp_path / "Cooking_and_Kitchen_Clean" / "clean_bowl" / "00001" / "00001")
    deep = _das_episode(
        tmp_path / "[STAGE 3]" / "domestic" / "bedroom" / "fold" / "store" / "0001" / "uuid"
    )
    orphan = _das_episode(tmp_path / "not_in_any_meta" / "00001" / "01706")
    nested = tmp_path / "Clutter Tidy-Up [Stage2]" / "00001" / "das_gripper_slim_meta.json"
    nested.write_text(json.dumps({"version": 1, "episodes": ["01706/episode.hdf5"]}))
    (tmp_path / "Clutter Tidy-Up [Stage2]" / "das_gripper_slim_meta.json").write_text(
        json.dumps({"version": 1, "episodes": ["00001/01706/episode.hdf5"]})
    )
    (tmp_path / "Cooking_and_Kitchen_Clean" / "das_gripper_slim_meta.json").write_text(
        json.dumps({"version": 1, "episodes": ["clean_bowl/00001/00001/episode.hdf5"]})
    )
    (tmp_path / "[STAGE 3]" / "das_gripper_slim_meta.json").write_text(
        json.dumps({"version": 1, "episodes": ["domestic/bedroom/fold/store/0001/uuid/episode.hdf5"]})
    )
    recs = scan(tmp_path, CUSTOM_SPECS["das_gripper"])
    assert {rec.path for rec in recs} == {str(clutter), str(cook), str(deep)}
    assert orphan.as_posix() not in {rec.path for rec in recs}
    by_path = {rec.path: rec for rec in recs}
    assert by_path[str(clutter)].lang == "Clutter Tidy-Up [Stage2]"
    assert by_path[str(cook)].lang == "clean bowl"
    assert by_path[str(deep)].lang == "store"
    limited = scan(tmp_path, CUSTOM_SPECS["das_gripper"], max_episodes=1)
    assert len(limited) == 1


def test_map_proc_reads_h5_nrows(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    from lbm.dataloader.custom.common.fs import h5_nframes, h5_nrows, map_proc

    path = tmp_path / "ep.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.zeros((5, 2), np.float32))
    jobs = [(str(path), "x")] * 20
    assert map_proc(h5_nrows, jobs) == [5] * 20
    assert h5_nframes([path] * 20, "x") == [5] * 20


def test_h5_nframes_process_pool(tmp_path: Path):
    """Pool uses fork; run outside pytest's threads so workers do not inherit them."""
    h5py = pytest.importorskip("h5py")
    import os
    import subprocess
    import sys

    path = tmp_path / "ep.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.zeros((5, 2), np.float32))
    script = tmp_path / "run.py"
    script.write_text(
        "from pathlib import Path\n"
        "from lbm.dataloader.custom.common.fs import h5_nframes\n"
        f"assert h5_nframes([Path({str(path)!r})] * 70, 'x') == [5] * 70\n"
    )
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_list_files_skips_hidden_and_keeps_siblings(tmp_path: Path):
    from lbm.dataloader.custom.common.fs import list_files

    real = tmp_path / "task" / "batch" / "ep"
    hidden = tmp_path / ".cache" / "batch" / "ep"
    real.mkdir(parents=True)
    hidden.mkdir(parents=True)
    (real / "episode.hdf5").write_bytes(b"x")
    (real / "cam_left_wrist.mp4").write_bytes(b"y")
    (hidden / "episode.hdf5").write_bytes(b"z")
    hits = list_files(tmp_path, name="episode.hdf5", dir_depth=3, siblings=True)
    assert [path for path, _names in hits] == [real / "episode.hdf5"]
    assert "cam_left_wrist.mp4" in hits[0][1]
    assert list_files(tmp_path, name="episode.hdf5", dir_depth=3, max_files=1) == [real / "episode.hdf5"]
    deep = tmp_path / "data" / "train" / "task" / "ep"
    deep.mkdir(parents=True)
    (deep / "episode.mcap").write_bytes(b"m")
    assert list_files(tmp_path / "data", name="episode.mcap") == [deep / "episode.mcap"]
    z1 = tmp_path / "aria" / "a.zarr"
    z2 = tmp_path / "aria" / "task" / "b.zarr"
    z1.mkdir(parents=True)
    z2.mkdir(parents=True)
    (tmp_path / "aria" / ".hidden.zarr").mkdir()
    from lbm.dataloader.custom.common.fs import list_dirs

    zarrs = list_dirs(tmp_path / "aria", suffix=".zarr", max_depth=2)
    assert zarrs == [z1, z2]


def test_scan_does_not_require_loading_all_videos(tmp_path):
    _v2_repo(
        tmp_path / "a",
        cams=("top_head", "hand_left", "hand_right"),
        state_col="observation.state",
        action_col="action",
        n=3,
    )
    recs = scan_root(tmp_path, CUSTOM_SPECS["kai0"])
    assert recs and recs[0].n_frames == 3
    assert recs[0].kind.startswith("lerobot")


def test_galaxea_split_columns(tmp_path):
    spec = CUSTOM_SPECS["galaxea"]
    repo = tmp_path / "Arrange_The_Fruits_20250822_013"
    meta = repo / "meta"
    data = repo / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 5
    (meta / "info.json").write_text(
        json.dumps(lerobot_info(version="v2.1", fps=15, robot_type="r1lite"))
    )
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["arrange"], "length": n}) + "\n")
    rows = {
        "observation.state.left_ee_pose": [np.ones(7, np.float32) for _ in range(n)],
        "observation.state.right_ee_pose": [np.ones(7, np.float32) for _ in range(n)],
        "observation.state.left_gripper": np.ones(n, np.float32),
        "observation.state.right_gripper": np.ones(n, np.float32),
        "observation.state.left_arm": [np.ones(7, np.float32) for _ in range(n)],
        "observation.state.right_arm": [np.ones(7, np.float32) for _ in range(n)],
        "observation.state.torso": [np.ones(2, np.float32) for _ in range(n)],
        "action.left_arm": [np.zeros(6, np.float32) for _ in range(n)],
        "action.right_arm": [np.zeros(6, np.float32) for _ in range(n)],
        "action.left_gripper": np.zeros(n, np.float32),
        "action.right_gripper": np.zeros(n, np.float32),
        "episode_index": np.zeros(n, np.int64),
        "frame_index": np.arange(n),
    }
    pd.DataFrame(rows).to_parquet(data / "episode_000000.parquet")
    for cam in spec.camera_keys:
        _mp4(repo / "videos" / "chunk-000" / f"observation.images.{cam}" / "episode_000000.mp4", n)
    ds = make_custom_dataset(tmp_path, "galaxea", data_cfg=custom_cfg())
    sample = ds[0]
    assert sample["state"].shape[-1] == 32
    assert sample["action"].shape[-1] == 14
    assert len(sample["image"]) == 3


def test_hifi_umi_nested_v3(tmp_path):
    root = tmp_path / "chunk-000" / "part-000"
    meta_ep = root / "meta" / "episodes" / "chunk-000"
    data = root / "data" / "chunk-000"
    meta_ep.mkdir(parents=True)
    data.mkdir(parents=True)
    n = 4
    (root / "meta" / "info.json").write_text(json.dumps(lerobot_info(version="v3.0", fps=25)))
    cams = ("head_main", "left_hand_up", "right_hand_up")
    ep_row = {
        "episode_index": 0,
        "length": n,
        "tasks": [["fold"]],
        "data/chunk_index": 0,
        "data/file_index": 0,
    }
    for cam in cams:
        key = f"observation.images.{cam}"
        ep_row[f"videos/{key}/chunk_index"] = 0
        ep_row[f"videos/{key}/file_index"] = 0
        ep_row[f"videos/{key}/from_timestamp"] = 0.0
        _mp4(root / "videos" / key / "chunk-000" / "file-000.mp4", n)
    pd.DataFrame([ep_row]).to_parquet(meta_ep / "file-000.parquet")
    rows = {
        "observation.state": [np.ones(20, np.float32) for _ in range(n)],
        "action": [np.zeros(20, np.float32) for _ in range(n)],
        "episode_index": np.zeros(n, np.int64),
        "frame_index": np.arange(n),
    }
    pd.DataFrame(rows).to_parquet(data / "file-000.parquet")
    ds = make_custom_dataset(tmp_path, "hifi_umi", data_cfg=custom_cfg())
    sample = ds[0]
    assert sample["state"].shape[-1] == 20
    assert sample["action"].shape[-1] == 20


def test_egoverse_zarr_layout(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "aria" / "ep0.zarr"
    store.parent.mkdir()
    t = 5
    zarr.create_array(str(store), name="left.obs_ee_pose", data=np.zeros((t, 7), np.float32))
    group = zarr.open(str(store), mode="r+")
    group["right.obs_ee_pose"] = np.ones((t, 7), np.float32)
    ds = make_custom_dataset(
        tmp_path / "aria", "egoverse", data_cfg=custom_cfg(use_mmap=True, use_mmap_frames=True)
    )
    sample = ds[0]
    assert sample["state"].shape[-1] == 16
    assert sample["action"].shape[-1] == 16
    assert np.allclose(sample["state"][:7], 0.0)
    assert sample["camera_mask"].tolist() == [True, False, False]
    assert int(np.asarray(sample["image"][1]).max()) == 0
    assert int(np.asarray(sample["image"][2]).max()) == 0
    assert list((tmp_path / "aria" / ".mmap").rglob("frames.bin")), "zarr stills should land in dump-root JPEG mmap"
    assert not list(store.rglob("frames.bin"))


def test_hy_lance_layout(tmp_path):
    lance = pytest.importorskip("lance")
    pa = pytest.importorskip("pyarrow")
    cv2 = pytest.importorskip("cv2")
    table_dir = tmp_path / "table_000"
    lance_path = table_dir / "table_000.lance"
    table_dir.mkdir()
    n = 6
    state = np.ones((n, 16), np.float32)
    action = np.stack([np.array([0.2, 0.8], np.float32) for _ in range(n)])
    ok, buf = cv2.imencode(".jpg", np.full((8, 8, 3), 80, dtype=np.uint8))
    assert ok
    jpeg = buf.tobytes()
    tbl = pa.table(
        {
            "observation_state": list(state),
            "action": list(action),
            "observation_images_cam_high": [jpeg] * n,
            "observation_images_cam_left_wrist": [jpeg] * n,
            "observation_images_cam_right_wrist": [jpeg] * n,
        }
    )
    lance.write_dataset(tbl, str(lance_path))
    meta = table_dir / "meta"
    meta.mkdir()
    (meta / "hy_episodes.jsonl").write_text(json.dumps({"episodes": [[0, 0, n]]}) + "\n")
    ds = make_custom_dataset(tmp_path, "hy_lance", data_cfg=custom_cfg(use_mmap=True, use_mmap_frames=True))
    sample = ds[0]
    assert sample["state"].shape[-1] == 16
    assert sample["action"].shape[-1] == 16
    assert abs(float(sample["action"][0, 7]) - 0.2) < 1e-5
    assert abs(float(sample["action"][0, 15]) - 0.8) < 1e-5
    assert sample["image"][0].shape[-1] == 3
    assert int(sample["image"][0].max()) > 0
    assert list((tmp_path / ".mmap").rglob("frames.bin")), "lance stills should land in dump-root JPEG mmap"
    assert not list(table_dir.rglob("frames.bin"))


def test_hy_lance_parquet_index_and_missing_images(tmp_path):
    lance = pytest.importorskip("lance")
    pa = pytest.importorskip("pyarrow")
    table_dir = tmp_path / "table_001"
    lance_path = table_dir / "table_001.lance"
    table_dir.mkdir()
    n = 4
    tbl = pa.table(
        {
            "observation_state": list(np.ones((n, 16), np.float32)),
            "action": list(np.ones((n, 2), np.float32)),
        }
    )
    lance.write_dataset(tbl, str(lance_path))
    ep_dir = table_dir / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": n, "length": n, "task_index": 0}]
    ).to_parquet(ep_dir / "file-000.parquet")
    ds = make_custom_dataset(tmp_path, "hy_lance", data_cfg=custom_cfg())
    sample = ds[0]
    assert sample["action"].shape[-1] == 16
    assert sample["image"][0].ndim == 4
    assert sample["image"][0].shape[-1] == 3


def test_mcap_annexb_decode():
    av = pytest.importorskip("av")
    from fractions import Fraction

    from lbm.dataloader.custom.datasets.abc import _decode_annexb

    try:
        codec = av.codec.CodecContext.create("libx264", "w")
    except Exception:
        pytest.skip("libx264 encoder not available")
    codec.width = 16
    codec.height = 16
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, 10)
    codec.options = {"preset": "ultrafast", "tune": "zerolatency"}
    try:
        codec.open()
    except Exception:
        pytest.skip("could not open libx264")
    packets: list[bytes] = []
    for i in range(4):
        rgb = np.full((16, 16, 3), (i + 1) * 40, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = i
        for packet in codec.encode(frame):
            packets.append(bytes(packet))
    for packet in codec.encode(None):
        packets.append(bytes(packet))
    if len(packets) < 1:
        pytest.skip("no h264 packets")
    decoded = _decode_annexb(packets, list(range(min(3, len(packets)))), "h264")
    assert decoded


def test_resolve_mmap_frames_mcap_and_agibot(tmp_path):
    from lbm.dataloader.custom.mmap_frames import collect_frame_refs, resolve_mmap_frames
    from lbm.dataloader.custom.record import EpisodeRecord

    dump = tmp_path / "dump"
    mcap = dump / "task" / "episode_0" / "episode.mcap"
    mcap.parent.mkdir(parents=True)
    mcap.write_bytes(b"not-a-real-mcap")
    rec = EpisodeRecord(kind="mcap", path=str(mcap), n_frames=8)
    ref = resolve_mmap_frames(rec, CUSTOM_SPECS["abc"], "cam_high", dump_root=dump)
    assert ref is not None
    assert ref.path == mcap
    assert ref.cache_root == dump
    assert ref.video_key == "cam_high"

    h5 = dump / "proprio_stats" / "1" / "2" / "x.h5"
    videos = dump / "observations" / "1" / "2" / "videos"
    videos.mkdir(parents=True)
    _mp4(videos / "head_color.mp4", 3)
    rec = EpisodeRecord(
        kind="agibot",
        path=str(h5),
        n_frames=3,
        extra={
            "video_dir": str(videos),
            "task_id": "1",
            "episode_id": "2",
            "videos": {"top_head": str(videos / "head_color.mp4")},
        },
    )
    ref = resolve_mmap_frames(rec, CUSTOM_SPECS["agibot"], "top_head", dump_root=dump)
    assert ref is not None and ref.kind == "mp4"
    assert ref.path.name == "head_color.mp4"
    assert ref.cache_root == dump

    refs = collect_frame_refs([rec], CUSTOM_SPECS["agibot"], dump_root=dump)
    assert [r.path.name for r in refs] == ["head_color.mp4"]
    assert refs[0].trajectory_id == ref.trajectory_id
    assert {r.cache_root for r in refs} == {dump}


def test_mmap_dump_root_groups_stores_and_dedupes_broadcast(tmp_path):
    from lbm.dataloader.custom.mmap_frames import _traj_id, collect_frame_refs
    from lbm.dataloader.custom.record import EpisodeRecord

    dump = tmp_path / "egoverse"
    z0 = dump / "a.zarr"
    z1 = dump / "b.zarr"
    z0.mkdir(parents=True)
    z1.mkdir(parents=True)
    recs = [
        EpisodeRecord(kind="zarr", path=str(z0), n_frames=4, extra={"video_key": "images.front_1"}),
        EpisodeRecord(kind="zarr", path=str(z1), n_frames=5, extra={"video_key": "images.front_1"}),
    ]
    refs = collect_frame_refs(recs, CUSTOM_SPECS["egoverse"], dump_root=dump)
    assert {r.cache_root for r in refs} == {dump}
    # 2 episodes × 1 shared stream (not 2 × 3 cameras)
    assert len(refs) == 2
    assert {r.video_key for r in refs} == {"images.front_1"}
    assert {r.cam for r in refs} == {"cam_high"}
    assert len({r.trajectory_id for r in refs}) == 2

    t0 = tmp_path / "table_000" / "table_000.lance"
    t1 = tmp_path / "table_001" / "table_001.lance"
    t0.mkdir(parents=True)
    t1.mkdir(parents=True)
    a = EpisodeRecord(kind="lance", path=str(t0), n_frames=3, extra={"start": 0, "episode_index": 0})
    b = EpisodeRecord(kind="lance", path=str(t1), n_frames=3, extra={"start": 0, "episode_index": 0})
    assert _traj_id(a) != _traj_id(b)
    lance_refs = collect_frame_refs([a, b], CUSTOM_SPECS["hy_lance"], dump_root=tmp_path)
    assert {r.cache_root for r in lance_refs} == {tmp_path}
    assert len(lance_refs) == 6



def test_contiguous_span():
    from lbm.dataloader.custom.video import contiguous_span

    assert contiguous_span([]) is None
    assert contiguous_span([3, 4, 5]) == (3, 6)
    assert contiguous_span([0, 1, 3]) is None
    assert contiguous_span([2, 1]) is None


def test_max_episodes_stops_scan(tmp_path):
    for name in ("a", "b"):
        _v2_repo(
            tmp_path / name,
            cams=("top_head", "hand_left", "hand_right"),
            state_col="observation.state",
            action_col="action",
            n=3,
        )
    recs = scan_root(tmp_path, CUSTOM_SPECS["kai0"], max_episodes=1)
    assert len(recs) == 1


def _write_abc_mcap(path: Path, *, t0: int, t1: int, camera_t0: int | None = None, camera_t1: int | None = None) -> None:
    from mcap.writer import Writer

    from lbm.dataloader.custom.datasets.abc import _SCALAR_TOPICS

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer = Writer(handle)
        writer.start(profile="", library="lbm-test")
        schema = writer.register_schema("empty", "jsonschema", b"{}")
        channels = [
            writer.register_channel(topic=topic, message_encoding="json", schema_id=schema)
            for topic in sorted(_SCALAR_TOPICS)
        ]
        for channel in channels:
            writer.add_message(channel, log_time=t0, data=b"{}", publish_time=t0)
            writer.add_message(channel, log_time=t1, data=b"{}", publish_time=t1)
        if camera_t0 is not None and camera_t1 is not None:
            cam = writer.register_channel(topic="/top-left-camera", message_encoding="json", schema_id=schema)
            writer.add_message(cam, log_time=camera_t0, data=b"{}", publish_time=camera_t0)
            writer.add_message(cam, log_time=camera_t1, data=b"{}", publish_time=camera_t1)
        writer.finish()


def test_abc_resampled_n_frames_matches_arange():
    from lbm.dataloader.custom.datasets.abc import _resampled_n_frames, _resample_ticks

    t0, t1, fps = 1775547359793520000, 1775547470908381696, 10.0
    ticks = _resample_ticks(t0, t1, fps)
    assert _resampled_n_frames(t0, t1, fps) == len(ticks)
    tick = int(round(1e9 / fps))
    np.testing.assert_array_equal(ticks, np.arange(t0, t1 + 1, tick, dtype=np.int64))


def test_abc_scan_n_frames_matches_scalar_overlap_not_camera_span(tmp_path: Path):
    pytest.importorskip("mcap")
    from lbm.dataloader.custom.datasets.abc import (
        _n_frames_mcap,
        _resampled_n_frames,
        _scalar_overlap_ns,
        scan,
    )

    mcap = tmp_path / "data" / "train" / "pick_up_the_cup" / "episode_0" / "episode.mcap"
    t0, t1 = 2_000_000_000, 3_000_000_000
    _write_abc_mcap(mcap, t0=t0, t1=t1, camera_t0=0, camera_t1=9_000_000_000)
    assert _scalar_overlap_ns(mcap) == (t0, t1)
    expect = _resampled_n_frames(t0, t1, 10.0)
    assert expect == 11
    assert _n_frames_mcap(mcap, 10.0) == expect
    recs = scan(tmp_path, CUSTOM_SPECS["abc"])
    assert recs[0].n_frames == expect
    assert recs[0].lang == "pick up the cup"


def test_abc_scan_walks_episode_mcap(tmp_path: Path):
    pytest.importorskip("mcap")
    from lbm.dataloader.custom.datasets.abc import _resampled_n_frames, scan

    mcap = tmp_path / "data" / "train" / "fold_the_towel" / "episode_x" / "episode.mcap"
    t0, t1 = 0, 2_000_000_000
    _write_abc_mcap(mcap, t0=t0, t1=t1)
    other = tmp_path / "data" / "train" / "pick_up_the_cup" / "episode_y" / "episode.mcap"
    _write_abc_mcap(other, t0=0, t1=5_000_000_000)
    recs = scan(tmp_path, CUSTOM_SPECS["abc"])
    by_path = {rec.path: rec for rec in recs}
    assert set(by_path) == {str(mcap), str(other)}
    assert by_path[str(mcap)].n_frames == _resampled_n_frames(t0, t1, 10.0) == 21
    assert by_path[str(mcap)].lang == "fold the towel"
    assert by_path[str(other)].lang == "pick up the cup"


def test_abc_scalar_overlap_when_camera_index_precedes_scalar(tmp_path: Path):
    """Index block starts with camera records; length covers the whole block."""
    pytest.importorskip("mcap")
    from mcap.writer import Writer

    from lbm.dataloader.custom.datasets.abc import _SCALAR_TOPICS, _scalar_overlap_ns

    path = tmp_path / "cam_first.mcap"
    t0, t1 = 1_000_000_000, 4_000_000_000
    with path.open("wb") as handle:
        writer = Writer(handle, chunk_size=256)
        writer.start(profile="", library="lbm-test")
        schema = writer.register_schema("empty", "jsonschema", b"{}")
        cam = writer.register_channel(topic="/top-left-camera", message_encoding="json", schema_id=schema)
        channels = [
            writer.register_channel(topic=topic, message_encoding="json", schema_id=schema)
            for topic in sorted(_SCALAR_TOPICS)
        ]
        for ts in (t0, t1):
            writer.add_message(cam, log_time=ts - 100, data=b"{}", publish_time=ts - 100)
            for channel in channels:
                writer.add_message(channel, log_time=ts, data=b"{}", publish_time=ts)
        writer.finish()
    assert _scalar_overlap_ns(path) == (t0, t1)


def test_abc_scalar_overlap_reads_only_endpoint_chunks(tmp_path: Path):
    pytest.importorskip("mcap")
    from mcap.writer import Writer

    from lbm.dataloader.custom.datasets.abc import _SCALAR_TOPICS, _scalar_overlap_ns

    path = tmp_path / "multi.mcap"
    t0, t1 = 1_000_000_000, 5_000_000_000
    mid = 3_000_000_000
    with path.open("wb") as handle:
        writer = Writer(handle, chunk_size=64)
        writer.start(profile="", library="lbm-test")
        schema = writer.register_schema("empty", "jsonschema", b"{}")
        channels = [
            writer.register_channel(topic=topic, message_encoding="json", schema_id=schema)
            for topic in sorted(_SCALAR_TOPICS)
        ]
        cam = writer.register_channel(topic="/top-left-camera", message_encoding="json", schema_id=schema)
        for ts in (t0, mid, t1):
            for channel in channels:
                writer.add_message(channel, log_time=ts, data=b"{}", publish_time=ts)
            writer.add_message(cam, log_time=ts + 50, data=b"{}", publish_time=ts + 50)
        writer.finish()
    assert _scalar_overlap_ns(path) == (t0, t1)


def test_abc_scan_skips_dot_cache_and_meta(tmp_path: Path):
    pytest.importorskip("mcap")
    from lbm.dataloader.custom.datasets.abc import scan

    real = tmp_path / "data" / "train" / "fold_the_towel" / "episode_x" / "episode.mcap"
    cached = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "download"
        / "data"
        / "train"
        / "fold_the_towel"
        / "episode_cached"
        / "episode.mcap"
    )
    meta = tmp_path / "meta" / "episode_meta" / "episode.mcap"
    _write_abc_mcap(real, t0=0, t1=2_000_000_000)
    _write_abc_mcap(cached, t0=0, t1=2_000_000_000)
    _write_abc_mcap(meta, t0=0, t1=2_000_000_000)
    recs = scan(tmp_path, CUSTOM_SPECS["abc"])
    assert [rec.path for rec in recs] == [str(real)]
