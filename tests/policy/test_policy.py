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


def test_mixed_checkpoint_keeps_model_io_and_adapts_source(tmp_path, monkeypatch):
    import torch
    from tests.helpers import tiny_dit_config

    from lbm.action_space import resolve_action_space
    from lbm.models.dit import DiTPolicy
    from lbm.policy import LBMPolicy
    from lbm.utils.preprocess import norm_stats_filename

    cfg = tiny_dit_config(camera_keys=('image', 'wrist_image', 'extra'),
                          state_dim=12, action_dim=10, action_length=2., action_freq=30.)
    trained = DiTPolicy(cfg)
    torch.save({'model': trained.state_dict()}, tmp_path/'model.pt')
    (tmp_path/'train_config.json').write_text(json.dumps(dict(model=asdict(cfg), flow={'num_diffusion_steps': 2})))
    stats = {key: dict(mean=[2.]*dim, std=[1.]*dim) for key, dim in [('state', 8), ('actions', 7)]}
    dump = tmp_path/'source'
    dump.mkdir()
    norm_name = norm_stats_filename(10., 2., slices=resolve_action_space(CUSTOM_SPECS['libero'], 'delta'))
    (dump/norm_name).write_text(json.dumps(dict(norm_stats=stats, action_freq=10.)))
    monkeypatch.setattr('lbm.policy.resolve_dataset', lambda *a, **kw: dump)
    policy = LBMPolicy.from_checkpoint(tmp_path/'model.pt', robot_type='libero', device='cpu')
    assert policy.model_config == cfg
    torch.testing.assert_close(policy.model.x_embedder.weight, trained.x_embedder.weight)
    assert policy.metadata()['state_dim'] == 8
    assert policy.metadata()['action_dim'] == 7
    assert policy.metadata()['chunk_length'] == 20
    assert policy.metadata()['camera_keys'] == ['image', 'wrist_image']
    monkeypatch.setattr(policy, '_task_vec', lambda prompt: torch.zeros(1, cfg.task_embed_dim))
    def sample(batch, **kwargs):
        assert batch['state'].shape == (1, 12)
        assert torch.count_nonzero(batch['state'][:, 8:]) == 0
        assert batch['camera_mask'].tolist() == [[True, True, False]]
        assert torch.count_nonzero(batch['images']['extra']) == 0
        return torch.zeros(1, 60, 10)
    monkeypatch.setattr(policy.model, 'sample_actions', sample)
    obs = dict(state=np.ones(8)*3, images={cam: np.zeros((16,16,3), dtype=np.uint8)
                                        for cam in ('image', 'wrist_image')}, reset=True)
    actions = policy.infer(obs)
    assert actions.shape == (20, 7) and np.isfinite(actions).all()
    prefix = policy.normalized_action_prefix(actions[:2], 2)
    assert prefix.shape == (60, 10)
    assert not prefix[:, 7:].any()
    assert not policy.normalized_action_prefix(np.empty((0, 7)), 0).any()
    with pytest.raises(KeyError, match='wrist_image'):
        policy.infer(dict(obs, images={'image': obs['images']['image']}))


def test_policy_supports_quantile_only_source_stats():
    import torch
    from tests.helpers import tiny_dit_config

    from lbm.policy import LBMPolicy

    cfg = tiny_dit_config(state_dim=12, action_dim=10)
    stats = {key: dict(q01=np.zeros(dim), q99=np.ones(dim)) for key, dim in [('state', 8), ('actions', 7)]}
    policy = LBMPolicy(None, config=cfg, device=torch.device('cpu'), norm_stats=stats)
    assert (policy.state_dim, policy.action_dim) == (8, 7)
