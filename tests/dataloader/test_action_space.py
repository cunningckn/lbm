import numpy as np
import pytest
from tests.fixtures.custom_cfg import custom_cfg

from lbm.action_space import (
    ABS,
    DELTA,
    REL,
    actions_from_state,
    apply_action_space,
    apply_delta_frames,
    bimanual_joint,
    derive_absolute_actions,
    invert_action_space,
    is_relative,
    parse_rep,
    resolve_action_space,
    unimanual_joint,
)
from lbm.dataloader.custom.datasets import CUSTOM_SPECS


def test_make_custom_dataset_requires_action_mode(tmp_path):
    from lbm.dataloader.custom import make_custom_dataset

    cfg = custom_cfg()
    del cfg["action_mode"]
    with pytest.raises(KeyError, match="action_mode"):
        make_custom_dataset(tmp_path, "kai0", data_cfg=cfg)


def test_parse_rep_aliases():
    assert parse_rep("absolute") == ABS
    assert parse_rep("relative") == REL
    assert parse_rep("delta") == DELTA
    with pytest.raises(ValueError, match="unknown rep"):
        parse_rep(None)
    with pytest.raises(ValueError, match="unknown rep"):
        parse_rep("absolute-delta")
    assert is_relative("rel") and is_relative("relative")
    assert not is_relative("delta") and not is_relative("abs")


def test_actions_from_state_abs_and_delta():
    state = np.arange(8, dtype=np.float32).reshape(8, 1)
    nxt = actions_from_state(state, native_fps=30.0, action_freq=10.0, mode=ABS)
    assert nxt[0, 0] == state[3, 0]
    assert nxt[-1, 0] == state[-1, 0]
    delta = actions_from_state(state, native_fps=30.0, action_freq=10.0, mode=DELTA)
    np.testing.assert_allclose(delta[0], state[3] - state[0])
    np.testing.assert_allclose(delta[-1], 0.0)
    rel = actions_from_state(state, native_fps=30.0, action_freq=10.0, mode="relative")
    np.testing.assert_allclose(rel, delta)


def test_bimanual_joint_roundtrip():
    slices = bimanual_joint(rep=REL)
    assert [s.name for s in slices] == ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    state = np.arange(14, dtype=np.float32)
    action = state + 1.0
    rel = apply_action_space(action[None], state, slices)[0]
    np.testing.assert_allclose(rel[0:6], 1.0)
    np.testing.assert_allclose(rel[6], action[6])
    np.testing.assert_allclose(rel[7:13], 1.0)
    back = invert_action_space(rel, state, slices)
    np.testing.assert_allclose(back, action)


def test_aloha_specs_default_delta_optional_rel():
    for name in ("kai0", "rmbench", "robotwin", "abc"):
        spec = CUSTOM_SPECS[name]
        delta = {s.name: s.rep for s in resolve_action_space(spec, DELTA)}
        assert delta["left_arm"] == DELTA
        assert delta["left_gripper"] == ABS
        rel = {s.name: s.rep for s in resolve_action_space(spec, REL)}
        assert rel["left_arm"] == REL
        assert rel["left_gripper"] == ABS


def test_unimanual_joint_roundtrip():
    slices = unimanual_joint(rep=REL)
    assert [s.name for s in slices] == ["arm", "gripper"]
    state = np.arange(8, dtype=np.float32)
    action = state + 1.0
    rel = apply_action_space(action[None], state, slices)[0]
    np.testing.assert_allclose(rel[0:7], 1.0)
    np.testing.assert_allclose(rel[7], action[7])
    back = invert_action_space(rel, state, slices)
    np.testing.assert_allclose(back, action)


def test_droid_default_delta_optional_rel():
    spec = CUSTOM_SPECS["droid"]
    delta = {s.name: s.rep for s in resolve_action_space(spec, DELTA)}
    assert delta["arm"] == DELTA
    assert delta["gripper"] == ABS
    rel = {s.name: s.rep for s in resolve_action_space(spec, REL)}
    assert rel["arm"] == REL
    assert sum(s.width for s in resolve_action_space(spec, DELTA)) == 8


def test_libero_eef_is_file_delta():
    space = resolve_action_space(CUSTOM_SPECS["libero"], REL)
    reps = {s.name: (s.rep, s.stored) for s in space}
    assert reps["eef"] == (DELTA, True)
    assert reps["gripper"] == (ABS, False)


def test_computed_delta_uses_action_length_hop():
    slices = bimanual_joint()
    action = np.arange(8 * 14, dtype=np.float32).reshape(8, 14)
    hop = 3
    out = apply_delta_frames(action, slices, hop)
    np.testing.assert_allclose(out[0, :6], action[3, :6] - action[0, :6])
    np.testing.assert_allclose(out[0, 6], action[0, 6])
    np.testing.assert_allclose(out[-1, :6], 0.0)


def test_libero_pins_action_freq_to_native_fps(caplog):
    from lbm.dataloader.custom import CustomSingleDataset, Episode

    spec = CUSTOM_SPECS["libero"]
    n = 8
    rng = np.random.default_rng(1)
    episode = Episode(
        images={cam: rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys},
        state=np.zeros((n, spec.state_dim), np.float32),
        action=np.zeros((n, spec.action_dim), np.float32),
    )
    with caplog.at_level("WARNING"):
        ds = CustomSingleDataset(spec, [episode], action_length=1.0, action_freq=5.0, action_mode=ABS)
    assert ds.action_freq == spec.fps
    np.testing.assert_array_equal(ds.action_deltas, np.arange(0, int(round(1.0 * spec.fps)), dtype=np.int64))
    assert "locked to 10" in caplog.text
    assert any("ignored 5" in rec.message for rec in caplog.records)


def test_derive_absolute_actions_uses_frequency_stride():
    spec = CUSTOM_SPECS["kai0"]
    state = np.arange(12 * 14, dtype=np.float32).reshape(12, 14)
    action = derive_absolute_actions(state, spec, native_fps=30.0, action_freq=10.0)
    np.testing.assert_allclose(action[0, :6], state[3, :6])
    np.testing.assert_allclose(action[-1], state[-1, :14])


def test_lerobot_without_action_derives_then_norms(tmp_path):
    from tests.fixtures.lerobot_tree import write_lerobot_v2_tree

    from lbm.dataloader.custom import make_custom_dataset
    from lbm.utils.preprocess import compute_norm_stats

    root = write_lerobot_v2_tree(tmp_path, robot_type="kai0", name="kai0", n_frames=12, include_action=False)
    ds = make_custom_dataset(root, "kai0", data_cfg=custom_cfg(action_freq=10.0, action_mode=DELTA))
    state, action = ds._vectors(0)
    expected = derive_absolute_actions(state, CUSTOM_SPECS["kai0"], native_fps=30.0, action_freq=10.0)
    np.testing.assert_allclose(action, expected, rtol=1e-5)
    payload = compute_norm_stats(ds, progress=False)
    assert payload["action_space"][0]["rep"] == DELTA
    assert payload["action_length"] == pytest.approx(ds.action_length)
    assert "q01" in payload["norm_stats"]["actions"]


def test_getitem_applies_rel_then_norm():
    from lbm.dataloader.custom import CustomSingleDataset, Episode

    spec = CUSTOM_SPECS["kai0"]
    rng = np.random.default_rng(0)
    n = 8
    episode = Episode(
        images={cam: rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys},
        state=np.linspace(0, 1, n * spec.state_dim, dtype=np.float32).reshape(n, spec.state_dim),
        action=np.linspace(1, 2, n * spec.action_dim, dtype=np.float32).reshape(n, spec.action_dim),
        lang="x",
    )
    ds = CustomSingleDataset(spec, [episode], action_length=0.2, action_mode=ABS)
    sample = ds[0]
    gathered = ds._gather_vector(episode.action, 0, ds.action_deltas, n)
    expected = apply_action_space(gathered, episode.state[0], resolve_action_space(spec, ABS))
    np.testing.assert_allclose(sample["action"], expected, rtol=1e-5)
