from tests.fixtures.robot_profiles import robot_io

from lbm.action_space import ABS, resolve_action_space
from lbm.dataloader.custom.datasets import CUSTOM_MIXTURES, CUSTOM_SPECS


def test_every_spec_has_required_fields():
    for name, spec in CUSTOM_SPECS.items():
        assert spec.camera_keys, name
        assert spec.state_dim > 0, name
        assert spec.action_dim > 0, name
        assert spec.embodiment, name
        assert spec.embodiment_id < 32, name
        io = robot_io(name)
        assert io.camera_keys == spec.camera_keys
        assert io.embodiment == spec.embodiment
        space = resolve_action_space(spec, ABS)
        assert space
        assert sum(sl.width for sl in space) == spec.action_dim


def test_rmbench_is_aloha():
    io = robot_io("rmbench")
    assert io.embodiment == "aloha"
    assert io.camera_keys == ("cam_high", "cam_left_wrist", "cam_right_wrist")


def test_libero_is_franka():
    io = robot_io("libero")
    assert io.embodiment == "franka"
    assert "image" in io.camera_keys
    assert "wrist_image" in io.camera_keys


def test_droid_is_oxe_droid():
    io = robot_io("droid")
    assert io.embodiment == "oxe_droid"
    assert io.camera_keys == ("exterior_1_left", "exterior_2_left", "wrist_left")
    assert io.state_dim == 8
    assert io.action_dim == 8


def test_mixture_specs_are_registered():
    known = set(CUSTOM_SPECS)
    for mix_name, spec in CUSTOM_MIXTURES.items():
        for _folder, _w, robot_type in spec:
            assert robot_type in known, f"{mix_name} uses unregistered {robot_type}"
