import numpy as np
import pytest
from tests.fixtures.custom_cfg import custom_cfg

from lbm.action_space import (
    ABS,
    DELTA,
    REL,
    XYZ_QUAT,
    XYZ_ROT6D,
    XYZ_ROTVEC,
    actions_from_state,
    actions_in_train_space,
    apply_action_space,
    bimanual_joint,
    derive_absolute_actions,
    eef_relative,
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


def test_computed_delta_is_consecutive_in_chunk():
    slices = bimanual_joint()
    action = np.arange(8 * 14, dtype=np.float32).reshape(8, 14)
    state = action[0].copy()
    chunk = action[[0, 3, 6]]
    out = actions_in_train_space(chunk, state, slices)
    np.testing.assert_allclose(out[0, :6], chunk[0, :6] - state[:6])
    np.testing.assert_allclose(out[1, :6], chunk[1, :6] - chunk[0, :6])
    np.testing.assert_allclose(out[0, 6], chunk[0, 6])
    back = invert_action_space(out, state, slices)
    np.testing.assert_allclose(back, chunk, atol=1e-5)


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


def _rand_eef(rng, fmt: str, n: int = 8) -> np.ndarray:
    from lbm.kinematics import matrix_to_pose, pose_to_matrix

    xyz = rng.normal(size=(n, 3)).astype(np.float64)
    rv = rng.normal(size=(n, 3)).astype(np.float64) * 0.4
    packed = np.concatenate([xyz, rv], axis=1)
    return matrix_to_pose(pose_to_matrix(packed, XYZ_ROTVEC), fmt)


@pytest.mark.parametrize("fmt", [XYZ_ROTVEC, XYZ_ROT6D, XYZ_QUAT])
def test_eef_relative_roundtrip(fmt):
    rng = np.random.default_rng(0)
    ref = _rand_eef(rng, fmt, 1)[0]
    act = _rand_eef(rng, fmt, 5)
    rel = eef_relative(act, ref, fmt)
    from lbm.action_space import eef_absolute

    back = eef_absolute(rel, ref, fmt)
    np.testing.assert_allclose(back, act, atol=1e-5)


def test_eef_consecutive_delta_differs_from_joint_and_rel():
    from lbm.action_space import EEF, GRIPPER, QUANTILE, ActionSlice
    from lbm.kinematics import invert44, matrix_to_pose, pose_to_matrix

    n = 6
    xyz = np.stack([np.linspace(0, 0.5, n), np.zeros(n), np.zeros(n)], axis=1)
    rv = np.zeros((n, 3))
    poses = np.concatenate([xyz, rv], axis=1).astype(np.float32)
    joint = np.linspace(0, 1, n * 6, dtype=np.float32).reshape(n, 6)
    eef_slices = (
        ActionSlice("eef", 0, 6, DELTA, EEF, QUANTILE, format=XYZ_ROTVEC),
        ActionSlice("gripper", 6, 7, ABS, GRIPPER, QUANTILE),
    )
    joint_slices = bimanual_joint()[:2]
    grip = np.ones((n, 1), np.float32)
    eef_packed = np.concatenate([poses, grip], axis=1)
    joint_packed = np.concatenate([joint, grip], axis=1)
    state_eef = eef_packed[0]
    state_joint = joint_packed[0]
    chunk_eef = eef_packed[[0, 2, 5]]
    chunk_joint = joint_packed[[0, 2, 5]]
    eef_delta = actions_in_train_space(chunk_eef, state_eef, eef_slices)
    joint_delta = actions_in_train_space(chunk_joint, state_joint, joint_slices)
    assert not np.allclose(eef_delta[:, :6], joint_delta[:, :6])
    eef_rel_slices = tuple(
        ActionSlice(s.name, s.start, s.end, REL, s.kind, s.norm, format=s.format) if s.kind != GRIPPER else s
        for s in eef_slices
    )
    eef_rel = actions_in_train_space(chunk_eef, state_eef, eef_rel_slices)
    far = np.linalg.norm(eef_rel[-1, :3])
    step = np.linalg.norm(eef_delta[-1, :3])
    assert far > step
    T_ref = pose_to_matrix(chunk_eef[0, :6], XYZ_ROTVEC)
    T_far = pose_to_matrix(chunk_eef[-1, :6], XYZ_ROTVEC)
    recovered = matrix_to_pose(invert44(T_ref) @ T_far, XYZ_ROTVEC)
    np.testing.assert_allclose(eef_rel[-1, :6], recovered, atol=1e-5)


def test_getitem_gather_then_delta_stays_in_window():
    from lbm.dataloader.custom import CustomSingleDataset, Episode

    spec = CUSTOM_SPECS["kai0"]
    n = 40
    action = np.arange(n * spec.action_dim, dtype=np.float32).reshape(n, spec.action_dim)
    state = action.copy()
    episode = Episode(
        images={cam: np.zeros((n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys},
        state=state,
        action=action,
        lang="x",
    )
    ds = CustomSingleDataset(spec, [episode], action_length=1.0, action_freq=10.0, action_mode=DELTA)
    sample = ds[0]
    last_native = int(ds.action_deltas[-1])
    assert last_native == 27 or last_native < n
    gathered = ds._gather_vector(action, 0, ds.action_deltas, n)
    expected = actions_in_train_space(gathered, state[0], resolve_action_space(spec, DELTA))
    np.testing.assert_allclose(sample["action"], expected, atol=1e-5)
    overshoot = action[min(last_native * 2, n - 1), :6] - action[0, :6]
    assert not np.allclose(sample["action"][-1, :6], overshoot)


def test_libero_stored_delta_is_not_converted():
    from lbm.dataloader.custom import CustomSingleDataset, Episode

    spec = CUSTOM_SPECS["libero"]
    n = 8
    rng = np.random.default_rng(2)
    action = rng.normal(size=(n, spec.action_dim)).astype(np.float32)
    episode = Episode(
        images={cam: rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys},
        state=rng.normal(size=(n, spec.state_dim)).astype(np.float32),
        action=action,
        lang="x",
    )
    ds = CustomSingleDataset(spec, [episode], action_length=0.5, action_mode=REL)
    sample = ds[0]
    gathered = ds._gather_vector(action, 0, ds.action_deltas, n)
    np.testing.assert_allclose(sample["action"][:, :6], gathered[:, :6])
    np.testing.assert_allclose(sample["action"][:, 6], gathered[:, 6])


def test_data_cfg_from_train_includes_kind_format():
    from lbm.config import TrainConfig, data_cfg_from_train

    cfg = TrainConfig()
    cfg.data.action_kind = "eef"
    cfg.data.action_format = "xyz+rot6d"
    data_cfg = data_cfg_from_train(cfg)
    assert data_cfg["action_kind"] == "eef"
    assert data_cfg["action_format"] == "xyz+rot6d"
    assert data_cfg["action_mode"] == "delta"


def test_kai0_eef_rot6d_norm_skips_rotation():
    from lbm.action_space import EEF, XYZ_ROT6D
    from lbm.dataloader.custom import CustomSingleDataset, Episode
    from lbm.utils.preprocess import normalize

    spec = CUSTOM_SPECS["kai0"]
    rng = np.random.default_rng(3)
    n = 8
    episode = Episode(
        images={cam: rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8) for cam in spec.camera_keys},
        state=rng.normal(size=(n, spec.state_dim)).astype(np.float32),
        action=rng.normal(size=(n, spec.action_dim)).astype(np.float32),
        lang="x",
    )
    ds = CustomSingleDataset(
        spec, [episode], action_length=0.2, action_mode=ABS, action_kind=EEF, action_format=XYZ_ROT6D
    )
    assert ds.policy_io["action_dim"] == 20
    state, action, slices = ds._policy_vectors(0)
    assert action.shape[-1] == 20
    stats = {
        "q01": np.zeros(20, np.float32),
        "q99": np.full(20, 2.0, np.float32),
    }
    gathered = ds._gather_vector(action, 0, ds.action_deltas, n)
    normed = normalize(gathered[0], stats, slices)
    np.testing.assert_allclose(normed[3:9], gathered[0, 3:9], atol=1e-5)
    assert not np.allclose(normed[:3], gathered[0, :3])
    rotvec_ds = CustomSingleDataset(
        spec, [episode], action_length=0.2, action_mode=ABS, action_kind=EEF, action_format=XYZ_ROTVEC
    )
    _, act_rv, slices_rv = rotvec_ds._policy_vectors(0)
    assert act_rv.shape[-1] == 14
    stats14 = {"q01": np.zeros(14, np.float32), "q99": np.full(14, 2.0, np.float32)}
    g_rv = rotvec_ds._gather_vector(act_rv, 0, rotvec_ds.action_deltas, n)
    norm_rv = normalize(g_rv[0], stats14, slices_rv)
    assert not np.allclose(norm_rv[3:6], g_rv[0, 3:6])
