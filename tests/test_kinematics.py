import numpy as np
import pytest

from lbm.action_space import DELTA, EEF, REL, bimanual_joint, resolve_action_space
from lbm.dataloader.custom.datasets import CUSTOM_SPECS
from lbm.kinematics import apply_joint_fk, joints_to_eef, lookup_chain, poe_fk, remap_joint_slices_to_eef
from lbm.utils.preprocess import norm_stats_filename

_CARTESIAN_DUMPS = ("libero", "hifi_umi", "hy_lance", "egoverse", "das_gripper")


@pytest.mark.parametrize("emb", ["aloha", "abc"])
def test_poe_zero_pose_matches_m(emb):
    chain = lookup_chain(emb, 6)
    t = poe_fk(np.zeros(6), chain)
    np.testing.assert_allclose(t[0], chain.m, atol=1e-6)
    pose = joints_to_eef(np.zeros(6), emb)
    np.testing.assert_allclose(pose[:3], chain.m[:3, 3], atol=1e-5)
    if emb == "aloha":
        np.testing.assert_allclose(pose[3:], 0.0, atol=1e-5)


def test_franka_zero_pose_uses_modified_dh():
    chain = lookup_chain("oxe_droid", 7)
    assert chain.modified
    pose = joints_to_eef(np.zeros(7), "oxe_droid")
    assert pose.shape == (6,)
    assert np.isfinite(pose).all()
    np.testing.assert_allclose(pose[:3], [0.088, 0.0, 0.926], atol=1e-4)


def test_g1_and_a1_zero_pose_are_finite():
    g1 = joints_to_eef(np.zeros(7), "agibot_genie1")
    a1 = joints_to_eef(np.zeros(6), "galaxea")
    assert g1.shape == a1.shape == (6,)
    assert np.isfinite(g1).all() and np.isfinite(a1).all()
    assert 0.4 < np.linalg.norm(g1[:3]) < 1.2
    assert 0.1 < np.linalg.norm(a1[:3]) < 1.2


def test_agibot_right_chain_negates_q2():
    left = lookup_chain("agibot_genie1", 7, side="left")
    right = lookup_chain("agibot_genie1", 7, side="right")
    assert left.name == "agibot_g1_left"
    assert right.name == "agibot_g1_right"
    assert left.joints[1].axis[2] == 1.0
    assert right.joints[1].axis[2] == -1.0
    q = np.zeros(7)
    q[1] = 0.4
    a = joints_to_eef(q, "agibot_genie1", side="left")
    b = joints_to_eef(q, "agibot_genie1", side="right")
    assert not np.allclose(a, b)


def test_missing_fk_raises():
    with pytest.raises(ValueError, match="no FK"):
        joints_to_eef(np.zeros(6), "no_such_robot")


def test_remap_joint_to_eef_renames_arms():
    space = remap_joint_slices_to_eef(bimanual_joint())
    reps = {s.name: s.kind for s in space}
    assert reps["left_eef"] == EEF
    assert reps["left_gripper"] == "gripper"


def test_kai0_fk_changes_values_and_kind():
    spec = CUSTOM_SPECS["kai0"]
    rng = np.random.default_rng(0)
    state = rng.normal(size=(4, spec.state_dim)).astype(np.float32)
    action = rng.normal(size=(4, spec.action_dim)).astype(np.float32)
    _st, act, slices = apply_joint_fk(state, action, spec, DELTA, EEF)
    assert {s.kind for s in slices if "eef" in s.name} == {EEF}
    assert not np.allclose(act[:, :6], action[:, :6])
    np.testing.assert_allclose(act[:, 6], action[:, 6])
    assert norm_stats_filename(10.0, 5.0, slices=slices) == "norm_stats_eef_delta_10hz_5s.json"


def test_resolve_action_kind_eef():
    space = resolve_action_space(CUSTOM_SPECS["kai0"], DELTA, action_kind=EEF)
    assert {s.name: s.kind for s in space}["left_eef"] == EEF


def test_agibot_fk_converts_arms_keeps_head():
    spec = CUSTOM_SPECS["agibot"]
    rng = np.random.default_rng(1)
    state = rng.normal(size=(3, spec.state_dim)).astype(np.float32)
    action = rng.normal(size=(3, spec.action_dim)).astype(np.float32)
    _st, act, slices = apply_joint_fk(state, action, spec, DELTA, EEF)
    kinds = {s.name: s.kind for s in slices}
    assert kinds["left_eef"] == EEF
    assert kinds["head"] == "joint"
    assert not np.allclose(act[:, :6], action[:, :6])
    np.testing.assert_allclose(act[:, 14:16], action[:, 14:16])
    np.testing.assert_allclose(act[:, 16:22], action[:, 16:22])


@pytest.mark.parametrize("name", _CARTESIAN_DUMPS)
def test_cartesian_dump_fk_is_noop(name):
    spec = CUSTOM_SPECS[name]
    state = np.ones((2, spec.state_dim), np.float32)
    action = np.full((2, spec.action_dim), 2.0, np.float32)
    mode = REL if name == "libero" else DELTA
    st, act, slices = apply_joint_fk(state, action, spec, mode, EEF)
    np.testing.assert_array_equal(st, state)
    np.testing.assert_array_equal(act, action)
    if name == "libero":
        assert any(s.stored for s in slices)
    else:
        assert all(s.kind != "joint" or s.name in {"head", "waist", "base"} for s in slices)


@pytest.mark.parametrize("name", sorted(CUSTOM_SPECS))
def test_every_spec_supports_action_kind_eef(name):
    spec = CUSTOM_SPECS[name]
    rng = np.random.default_rng(2)
    state = rng.normal(size=(2, spec.state_dim)).astype(np.float32)
    action = rng.normal(size=(2, spec.action_dim)).astype(np.float32)
    st, act, slices = apply_joint_fk(state, action, spec, DELTA, EEF)
    assert st.shape == state.shape
    assert act.shape == action.shape
    assert slices
