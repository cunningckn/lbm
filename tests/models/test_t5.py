import pytest
import torch

from lbm.config import DiTConfig
from lbm.models.t5 import (
    T5LanguageEncoder,
    T5LayerNorm,
    _additive_pad_mask,
    _relative_position_bucket,
)


def _encoder(**kwargs) -> T5LanguageEncoder:
    defaults = dict(
        t5_vocab_size=64,
        t5_d_model=32,
        t5_d_ff=64,
        t5_num_layers=2,
        t5_num_heads=4,
    )
    defaults.update(kwargs)
    return T5LanguageEncoder(DiTConfig(**defaults))


def test_t5_forward_shape_and_out_dim():
    enc = _encoder()
    ids = torch.randint(1, 64, (3, 8))
    out = enc(ids)
    assert enc.out_dim == 32
    assert out.shape == (3, 8, 32)
    assert torch.isfinite(out).all()


def test_t5_padding_mask_ignores_pad_tokens():
    enc = _encoder().eval()
    ids = torch.tensor([[4, 5, 6, 0, 0], [4, 5, 6, 31, 31]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        a = enc(ids[:1], attention_mask=mask[:1])
        b = enc(ids[1:], attention_mask=mask[1:])
    torch.testing.assert_close(a[:, :3], b[:, :3], rtol=1e-5, atol=1e-5)


def test_t5_default_mask_treats_zero_as_pad():
    enc = _encoder().eval()
    ids = torch.tensor([[7, 8, 9, 0, 0]])
    with torch.no_grad():
        implicit = enc(ids)
        explicit = enc(ids, attention_mask=(ids != 0).float())
    torch.testing.assert_close(implicit, explicit)


def test_additive_pad_mask_matches_hf_finfo_min():
    keep = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = _additive_pad_mask(keep, torch.float32)
    assert mask.shape == (1, 1, 1, 4)
    torch.testing.assert_close(mask[..., :2], torch.zeros(1, 1, 1, 2))
    torch.testing.assert_close(
        mask[..., 2:],
        torch.full((1, 1, 1, 2), torch.finfo(torch.float32).min),
    )
    half = _additive_pad_mask(keep, torch.float16)
    torch.testing.assert_close(
        half[..., 2:],
        torch.full((1, 1, 1, 2), torch.finfo(torch.float16).min, dtype=torch.float16),
    )


def test_t5_dropout_active_in_train_disabled_in_eval():
    enc = _encoder()
    ids = torch.randint(1, 64, (2, 8))
    enc.train()
    torch.manual_seed(0)
    a = enc(ids)
    torch.manual_seed(0)
    b = enc(ids)
    torch.testing.assert_close(a, b)
    torch.manual_seed(1)
    c = enc(ids)
    assert not torch.allclose(a, c)
    enc.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        d = enc(ids)
        torch.manual_seed(1)
        e = enc(ids)
    torch.testing.assert_close(d, e)


def test_t5_relative_bias_only_on_first_layer():
    enc = _encoder(t5_num_layers=3)
    assert enc.layers[0].self_attn.has_relative_bias
    assert hasattr(enc.layers[0].self_attn, "relative_attention_bias")
    assert not enc.layers[1].self_attn.has_relative_bias
    assert not enc.layers[2].self_attn.has_relative_bias


def test_t5_backward():
    enc = _encoder()
    ids = torch.randint(1, 64, (2, 6))
    enc(ids).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


def test_t5_layer_norm_rms():
    ln = T5LayerNorm(4)
    x = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    out = ln(x)
    torch.testing.assert_close(out, torch.tensor([[2.0, 0.0, 0.0, 0.0]]))


def test_relative_position_bucket_range():
    rel = torch.arange(-40, 41)
    buckets = _relative_position_bucket(rel)
    assert buckets.dtype == torch.long
    assert int(buckets.min()) >= 0
    assert int(buckets.max()) < 32
    assert int(_relative_position_bucket(torch.tensor([0]))[0]) == 0


def _hf_t5_state(vocab=64, d_model=32, d_ff=64, layers=2, heads=4):
    state = {
        "shared.weight": torch.randn(vocab, d_model),
        "encoder.final_layer_norm.weight": torch.ones(d_model),
        "decoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(d_model, d_model),
    }
    for i in range(layers):
        src = f"encoder.block.{i}"
        for name in ("q", "k", "v", "o"):
            state[f"{src}.layer.0.SelfAttention.{name}.weight"] = torch.randn(d_model, d_model)
        if i == 0:
            state[f"{src}.layer.0.SelfAttention.relative_attention_bias.weight"] = torch.randn(32, heads)
        state[f"{src}.layer.0.layer_norm.weight"] = torch.ones(d_model)
        state[f"{src}.layer.1.DenseReluDense.wi.weight"] = torch.randn(d_ff, d_model)
        state[f"{src}.layer.1.DenseReluDense.wo.weight"] = torch.randn(d_model, d_ff)
        state[f"{src}.layer.1.layer_norm.weight"] = torch.ones(d_model)
    return state


def test_convert_hf_t5_drops_decoder_and_maps_relative_bias():
    from lbm.models.t5 import convert_hf_t5

    hf = _hf_t5_state()
    native = convert_hf_t5(hf)
    assert not any(k.startswith("decoder.") for k in native)
    torch.testing.assert_close(native["embed_tokens.weight"], hf["shared.weight"])
    torch.testing.assert_close(
        native["layers.0.self_attn.relative_attention_bias.weight"],
        hf["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
    )
    assert "layers.1.self_attn.relative_attention_bias.weight" not in native


def test_load_t5_hf_and_native(tmp_path):
    from lbm.models.t5 import load_t5

    enc = _encoder()
    hf_path = tmp_path / "t5_hf.pt"
    torch.save(_hf_t5_state(), hf_path)
    load_t5(enc, hf_path)
    ids = torch.randint(1, 64, (2, 6))
    assert torch.isfinite(enc(ids)).all()

    native_path = tmp_path / "t5_native.pt"
    torch.save(enc.state_dict(), native_path)
    clone = _encoder()
    load_t5(clone, native_path)
    torch.testing.assert_close(clone.embed_tokens.weight, enc.embed_tokens.weight)


def test_load_real_t5_if_present():
    from lbm.models.t5 import load_t5, resolve_t5_path

    path = resolve_t5_path()
    if path is None:
        pytest.skip("no T5 checkpoint")
    enc = T5LanguageEncoder(DiTConfig(language_encoder="t5"))
    load_t5(enc, path)
    ids = torch.randint(1, 100, (1, 8))
    out = enc(ids)
    assert out.shape == (1, 8, 512)
    assert torch.isfinite(out).all()
