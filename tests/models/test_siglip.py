import pytest
import torch

from lbm.config import DiTConfig
from lbm.models.siglip import SiglipVisionBackbone, SiglipVisionTransformer


def _cfg(**kwargs) -> DiTConfig:
    defaults = dict(vit_embed_dim=32, vit_depth=2, vit_num_heads=4)
    defaults.update(kwargs)
    return DiTConfig(**defaults)


def test_siglip_tokens_have_no_cls():
    backbone = SiglipVisionBackbone(_cfg())
    images = torch.randn(2, 3, 224, 224)
    tokens = backbone.encode_image_tokens(images)
    patches = (224 // 16) ** 2
    assert tokens.shape == (2, patches, 32)
    assert torch.isfinite(tokens).all()


def test_siglip_pos_embed_interpolates_for_non_224():
    vit = SiglipVisionTransformer(embed_dim=32, depth=1, num_heads=4, image_size=224, patch_size=16)
    images = torch.randn(1, 3, 32, 32)
    tokens = vit.encode_image_tokens(images)
    assert tokens.shape == (1, (32 // 16) ** 2, 32)
    assert torch.isfinite(tokens).all()


def test_siglip_backward():
    backbone = SiglipVisionBackbone(_cfg())
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    loss = backbone.encode_image_tokens(images).sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())


def test_siglip_set_bfloat16_returns_fp32(device):
    if device.type != "cuda":
        pytest.skip("bf16 autocast is CUDA-only")
    backbone = SiglipVisionBackbone(_cfg()).to(device)
    backbone.set_bfloat16(True)
    images = torch.randn(1, 3, 32, 32, device=device)
    tokens = backbone.encode_image_tokens(images)
    assert tokens.dtype == torch.float32
    assert torch.isfinite(tokens).all()


def _hf_siglip_state(dim=32, depth=2, heads=4, patches=4):
    state = {
        "vision_model.embeddings.patch_embedding.weight": torch.randn(dim, 3, 16, 16),
        "vision_model.embeddings.patch_embedding.bias": torch.randn(dim),
        "vision_model.embeddings.position_embedding.weight": torch.randn(patches, dim),
        "vision_model.post_layernorm.weight": torch.ones(dim),
        "vision_model.post_layernorm.bias": torch.zeros(dim),
        "text_model.embeddings.token_embedding.weight": torch.randn(8, dim),
    }
    for i in range(depth):
        prefix = f"vision_model.encoder.layers.{i}"
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            state[f"{prefix}.self_attn.{name}.weight"] = torch.randn(dim, dim)
            state[f"{prefix}.self_attn.{name}.bias"] = torch.randn(dim)
        for name in ("layer_norm1", "layer_norm2"):
            state[f"{prefix}.{name}.weight"] = torch.ones(dim)
            state[f"{prefix}.{name}.bias"] = torch.zeros(dim)
        state[f"{prefix}.mlp.fc1.weight"] = torch.randn(dim * 4, dim)
        state[f"{prefix}.mlp.fc1.bias"] = torch.randn(dim * 4)
        state[f"{prefix}.mlp.fc2.weight"] = torch.randn(dim, dim * 4)
        state[f"{prefix}.mlp.fc2.bias"] = torch.randn(dim)
    return state


def test_convert_hf_siglip_fuses_qkv_and_drops_text():
    from lbm.models.siglip import convert_hf_siglip

    hf = _hf_siglip_state()
    native = convert_hf_siglip(hf)
    assert "text_model.embeddings.token_embedding.weight" not in native
    q = hf["vision_model.encoder.layers.0.self_attn.q_proj.weight"]
    k = hf["vision_model.encoder.layers.0.self_attn.k_proj.weight"]
    v = hf["vision_model.encoder.layers.0.self_attn.v_proj.weight"]
    torch.testing.assert_close(native["blocks.0.attn.qkv.weight"], torch.cat([q, k, v], dim=0))
    assert native["pos_embed"].shape == (1, 4, 32)


def test_load_siglip_hf_and_native(tmp_path):
    from lbm.models.siglip import load_siglip

    vit = SiglipVisionTransformer(embed_dim=32, depth=2, num_heads=4, image_size=32, patch_size=16)
    hf_path = tmp_path / "siglip_hf.pt"
    torch.save(_hf_siglip_state(), hf_path)
    load_siglip(vit, hf_path)
    images = torch.randn(1, 3, 32, 32)
    assert torch.isfinite(vit.encode_image_tokens(images)).all()

    native_path = tmp_path / "siglip_native.pt"
    torch.save(vit.state_dict(), native_path)
    clone = SiglipVisionTransformer(embed_dim=32, depth=2, num_heads=4, image_size=32, patch_size=16)
    load_siglip(clone, native_path)
    torch.testing.assert_close(clone.pos_embed, vit.pos_embed)


def test_load_real_siglip_if_present():
    from lbm.models.siglip import load_siglip, resolve_siglip_path

    path = resolve_siglip_path()
    if path is None:
        pytest.skip("no SigLIP checkpoint")
    backbone = SiglipVisionBackbone(DiTConfig())
    load_siglip(backbone, path)
    tokens = backbone.encode_image_tokens(torch.randn(1, 3, 224, 224))
    assert tokens.shape == (1, 196, 768)
    assert torch.isfinite(tokens).all()
