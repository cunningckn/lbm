import copy

import pytest
import torch
from tests.helpers import tiny_dit_config

from lbm.models.dit import DiTPolicy
from lbm.utils.fake_data import make_fake_batch


def test_optional_state_history_preserves_shared_initialization_and_initial_predictions():
    config = tiny_dit_config()
    torch.manual_seed(123)
    baseline = DiTPolicy(config).eval()
    rng = torch.get_rng_state()
    enabled = copy.deepcopy(config)
    enabled.state_history_length = 0.3
    torch.manual_seed(123)
    candidate = DiTPolicy(enabled).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])
    batch = make_fake_batch(config, 2, device="cpu", image_size=32)
    batch.update(
        state_history=torch.randn(2, 3, config.state_dim),
        state_history_mask=torch.ones(2, 3, dtype=torch.bool),
        state_history_offsets=torch.tensor([[-0.2, -0.1, 0.0]]).expand(2, -1),
    )
    noise = torch.randn_like(batch["actions"])
    torch.testing.assert_close(
        baseline.sample_actions(batch, num_steps=2, noise=noise),
        candidate.sample_actions(batch, num_steps=2, noise=noise),
        rtol=0,
        atol=0,
    )
    candidate.state_history_pool.gate.data.fill_(1.0)
    before = candidate.encode_state_history(batch)
    batch["state_history"] = batch["state_history"].flip(1)
    after = candidate.encode_state_history(batch)
    assert not torch.equal(before, after)
    batch["state_history_mask"][:, 0] = False
    before = candidate.encode_state_history(batch)
    batch["state_history"][:, 0] = float("nan")
    torch.testing.assert_close(candidate.encode_state_history(batch), before, rtol=0, atol=0)


def test_timed_vision_masks_invalid_values_and_distinguishes_order():
    config = tiny_dit_config()
    config.history_time_encoding = True
    model = DiTPolicy(config).eval()
    b, t, patches = 2, 3, 4
    features = {cam: torch.randn(b, t, patches, config.vit_embed_dim) for cam in config.camera_keys}
    valid = torch.tensor([[False, True, True]]).expand(b, -1)
    offsets = torch.tensor([[-0.2, -0.1, 0.0]]).expand(b, -1)
    tokens, mask = model.build_vision_tokens(features=features, history_mask=valid, history_offsets=offsets)
    query = torch.randn(b, 2, config.hidden_size)
    attention = model.blocks[0].cross_attn
    before = attention(query, tokens, mask)
    for feature in features.values():
        feature[:, 0] = float("nan")
    tokens, mask = model.build_vision_tokens(features=features, history_mask=valid, history_offsets=offsets)
    torch.testing.assert_close(attention(query, tokens, mask), before, rtol=0, atol=0)
    for feature in features.values():
        feature[:, 1:] = feature[:, 1:].flip(1)
    tokens, mask = model.build_vision_tokens(features=features, history_mask=valid, history_offsets=offsets)
    assert not torch.equal(attention(query, tokens, mask), before)
    with pytest.raises(ValueError, match="nonpositive"):
        model.build_vision_tokens(features=features, history_mask=valid, history_offsets=-offsets)


def test_masked_camera_nan_does_not_affect_timed_vision_gradients():
    config = tiny_dit_config()
    config.history_time_encoding = True
    model = DiTPolicy(config)
    batch = make_fake_batch(config, 2, image_size=32)
    cameras = torch.ones(2, len(config.camera_keys), dtype=torch.bool)
    cameras[:, -1] = False
    kw = dict(camera_mask=cameras, history_mask=batch["history_mask"], history_offsets=batch["history_offsets"])
    clean, mask = model.build_vision_tokens(batch["images"], **kw)
    batch["images"][config.camera_keys[-1]].fill_(float("nan"))
    dirty, other_mask = model.build_vision_tokens(batch["images"], **kw)
    torch.testing.assert_close(dirty, clean, rtol=0, atol=0)
    assert torch.equal(mask, other_mask)
    query = torch.randn(2, 2, config.hidden_size)
    result = model.blocks[0].cross_attn(query, dirty, mask)
    result.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.gpu
def test_history_initialization_preserves_rng_when_constructed_directly_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cfg = tiny_dit_config()
    with torch.device("cuda"):
        torch.manual_seed(73)
        baseline = DiTPolicy(cfg)
        rng = torch.cuda.get_rng_state()
        cfg.state_history_length = 0.3
        torch.manual_seed(73)
        candidate = DiTPolicy(cfg)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])
