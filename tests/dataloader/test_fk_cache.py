"""Joint→EEF proprio cache at ``{root}/.cache/fk``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lbm.action_space import EEF
from lbm.dataloader.custom import CUSTOM_SPECS, CustomSingleDataset, Episode, save_numpy_episode
from lbm.dataloader.custom.fk_cache import (
    cache_dir,
    episode_dir,
    load_fk_episode,
    needs_joint_fk,
    prebuild_fk,
    source_key,
    write_fk_episode,
)
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.kinematics import apply_joint_fk


def _episode(spec, n_frames: int = 8, *, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed)
    images = {cam: rng.integers(0, 255, size=(n_frames, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys}
    return Episode(
        images=images,
        state=rng.standard_normal((n_frames, spec.state_dim)).astype(np.float32),
        action=rng.standard_normal((n_frames, spec.action_dim)).astype(np.float32),
        lang="pick",
    )


def _numpy_dataset(tmp_path: Path, spec, *, action_kind: str | None = EEF) -> CustomSingleDataset:
    ep = _episode(spec)
    path = tmp_path / "ep0.npz"
    save_numpy_episode(path, ep)
    rec = EpisodeRecord(kind="numpy", path=str(path), n_frames=int(ep.action.shape[0]))
    return CustomSingleDataset(
        spec,
        records=[rec],
        action_mode="delta",
        action_kind=action_kind,
        root=tmp_path,
    )


def test_needs_joint_fk_matches_registered_arms():
    assert needs_joint_fk(CUSTOM_SPECS["kai0"])
    assert needs_joint_fk(CUSTOM_SPECS["agibot"])
    assert not needs_joint_fk(CUSTOM_SPECS["libero"])
    assert not needs_joint_fk(CUSTOM_SPECS["hifi_umi"])


def test_write_load_roundtrip(tmp_path: Path):
    state = np.arange(24, dtype=np.float32).reshape(4, 6)
    action = state + 1.0
    dest = episode_dir(tmp_path, 0)
    write_fk_episode(dest, state, action, source="/ep0", n_frames=4)
    got = load_fk_episode(tmp_path, 0, source="/ep0", n_frames=4)
    assert got is not None
    np.testing.assert_array_equal(got[0], state)
    np.testing.assert_array_equal(got[1], action)
    assert load_fk_episode(tmp_path, 0, source="/other", n_frames=4) is None
    assert load_fk_episode(tmp_path, 0, source="/ep0", n_frames=3) is None


def test_prebuild_skips_cartesian_dump(tmp_path: Path):
    ds = _numpy_dataset(tmp_path, CUSTOM_SPECS["libero"])
    built, reused = prebuild_fk(ds, workers=1)
    assert built == reused == 0
    assert not cache_dir(tmp_path).is_dir()


def test_prebuild_and_policy_vectors_use_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = CUSTOM_SPECS["kai0"]
    ds = _numpy_dataset(tmp_path, spec)
    rec = ds.records[0]
    raw_state, raw_action = ds._read_vectors(rec)
    expect_st, expect_act, expect_slices = apply_joint_fk(raw_state, raw_action, spec, "delta", EEF)

    ds.prebuild_fk_cache(workers=1)
    dest = episode_dir(tmp_path, 0)
    man = json.loads((dest / "manifest.json").read_text())
    assert man["source"] == source_key(rec)
    assert man["n_frames"] == rec.n_frames
    assert man["version"] == 1
    assert man["pose_format"] == "xyz_rotvec"
    assert (cache_dir(tmp_path) / "manifest.json").is_file()

    def boom(*_a, **_k):
        raise AssertionError("live FK should not run when cache exists")

    monkeypatch.setattr("lbm.kinematics.apply_joint_fk", boom)
    st, act, slices = ds._policy_vectors(0)
    np.testing.assert_allclose(st, expect_st, atol=1e-5)
    np.testing.assert_allclose(act, expect_act, atol=1e-5)
    assert [s.name for s in slices] == [s.name for s in expect_slices]


def test_prebuild_reuses_existing_episodes(tmp_path: Path):
    ds = _numpy_dataset(tmp_path, CUSTOM_SPECS["kai0"])
    assert prebuild_fk(ds, workers=1) == (1, 0)
    assert prebuild_fk(ds, workers=1) == (0, 1)
    assert prebuild_fk(ds, workers=1, force=True) == (1, 0)


def test_fk_v1_rotvec_cache_loads_as_rot6d(tmp_path: Path):
    from lbm.action_space import XYZ_ROT6D

    spec = CUSTOM_SPECS["kai0"]
    ds = _numpy_dataset(tmp_path, spec)
    ds.prebuild_fk_cache(workers=1)
    ds6 = CustomSingleDataset(
        spec,
        records=ds.records,
        action_mode="delta",
        action_kind=EEF,
        action_format=XYZ_ROT6D,
        root=tmp_path,
    )
    st, act, slices = ds6._policy_vectors(0)
    assert act.shape[-1] == 20
    assert st.shape[-1] == 20
    assert slices[0].width == 9
    assert slices[0].format == XYZ_ROT6D
    dest = episode_dir(tmp_path, 0)
    man = json.loads((dest / "manifest.json").read_text())
    assert man["version"] == 1
    assert man["pose_format"] == "xyz_rotvec"
