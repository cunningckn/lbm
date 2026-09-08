import json
from dataclasses import asdict

import numpy as np
import pytest

from lbm.config import DiTConfig, TrainConfig
from lbm.dataloader.custom.datasets import CUSTOM_SPECS
from lbm.policy import apply_spec, dit_config_from_mapping, load_train_model_config
from lbm.policy_codec import decode_array, decode_obs, encode_array, encode_obs


def test_dit_config_from_train_json_mapping():
    raw = asdict(DiTConfig(camera_keys=("image", "wrist_image"), state_dim=8, action_dim=7, action_freq=10.0))
    cfg = dit_config_from_mapping(raw)
    assert cfg.camera_keys == ("image", "wrist_image")
    assert cfg.state_dim == 8
    assert cfg.action_dim == 7
    assert cfg.action_freq == 10.0
    assert cfg.chunk_length == 50


def test_load_train_model_config(tmp_path):
    train = TrainConfig()
    train.model.camera_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    train.flow.num_diffusion_steps = 7
    path = tmp_path / "train_config.json"
    path.write_text(json.dumps(asdict(train)))
    cfg, steps = load_train_model_config(path)
    assert cfg.camera_keys == ("cam_high", "cam_left_wrist", "cam_right_wrist")
    assert steps == 7


def test_apply_spec_libero_and_rmbench():
    libero = apply_spec(DiTConfig(), "libero")
    spec = CUSTOM_SPECS["libero"]
    assert libero.camera_keys == spec.camera_keys
    assert libero.state_dim == spec.state_dim
    assert libero.action_dim == spec.action_dim
    assert libero.action_freq == spec.fps

    rmbench = apply_spec(DiTConfig(camera_keys=("top",)), "rmbench")
    assert rmbench.camera_keys == CUSTOM_SPECS["rmbench"].camera_keys
    assert rmbench.state_dim == 14
    assert rmbench.action_freq == CUSTOM_SPECS["rmbench"].fps


def test_apply_spec_unknown_raises():
    cfg = DiTConfig(state_dim=3)
    with pytest.raises(KeyError):
        apply_spec(cfg, "not_a_robot")


def test_encode_decode_array_roundtrip():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = decode_array(encode_array(x))
    assert y.dtype == np.float32
    assert y.shape == (3, 4)
    np.testing.assert_array_equal(y, x)


def test_encode_decode_uint8_image():
    img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    out = decode_array(encode_array(img))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, img)


def test_obs_payload_roundtrip():
    obs = {
        "state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "images": {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
        },
        "prompt": "pick up the cup",
        "reset": True,
    }
    restored = decode_obs(encode_obs(obs))
    np.testing.assert_array_equal(restored["state"], obs["state"])
    np.testing.assert_array_equal(restored["images"]["image"], obs["images"]["image"])
    assert restored["prompt"] == "pick up the cup"
    assert restored["reset"] is True
