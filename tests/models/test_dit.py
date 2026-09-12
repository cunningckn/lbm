import pytest
import torch

from lbm.models.clip import task_name_to_prompt


def test_task_name_to_prompt_replaces_separators():
    assert task_name_to_prompt("open_the-pen_caps") == "open the pen caps"


def test_clip_encodes_prompt(clip, config, task_vec):
    assert task_vec.shape == (1, config.task_embed_dim)
    assert torch.isfinite(task_vec).all()
    again = clip.encode("put the bottles in the bin")
    torch.testing.assert_close(again, task_vec)


def test_clip_set_bfloat16_returns_fp32(clip, device):
    if device.type != "cuda":
        pytest.skip("bf16 autocast is CUDA-only")
    prev = clip.bfloat16
    clip.set_bfloat16(True)
    try:
        out = clip.encode("bf16 autocast probe")
    finally:
        clip.set_bfloat16(prev)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_clip_attn_matches_mha(device):
    """``attention()`` causal path matches ``nn.MultiheadAttention`` + CLIP mask."""
    from lbm.models.clip import _mha_self_attn

    torch.manual_seed(0)
    width, heads, seq, batch = 512, 8, 77, 2
    attn = torch.nn.MultiheadAttention(width, heads).to(device).eval()
    x = torch.randn(seq, batch, width, device=device)
    mask = torch.empty(seq, seq, device=device).fill_(float("-inf")).triu_(1)
    with torch.no_grad():
        ref = attn(x, x, x, need_weights=False, attn_mask=mask)[0]
        out = _mha_self_attn(attn, x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_policy_uses_bf16_on_cuda(model, batch, device, dtype):
    assert next(model.parameters()).dtype == dtype
    assert batch["state"].dtype == dtype
    if device.type == "cuda":
        assert dtype == torch.bfloat16


def test_dino_encodes_image_tokens(model, config, batch):
    cam = config.camera_keys[0]
    tokens = model.img_backbone.encode_image_tokens(batch["images"][cam])
    patches = (224 // 16) ** 2
    assert tokens.shape == (1, 1 + patches, config.vit_embed_dim)
    assert torch.isfinite(tokens).all()


def test_load_dinov3_hf_safetensors(config, dinov3_path):
    from lbm import load_dinov3
    from lbm.models.dino import DinoVisionBackbone

    backbone = DinoVisionBackbone(config)
    load_dinov3(backbone, dinov3_path)
    qkv_bias = backbone.dinov3_model.blocks[0].attn.qkv.bias.detach()
    dim = config.vit_embed_dim
    assert torch.count_nonzero(qkv_bias[dim : 2 * dim]) == 0



def test_policy_forward_and_sample(model, config, batch):
    model.eval()
    loss = model(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    actions = model.sample_actions(batch, num_steps=2)
    assert actions.shape == (1, config.chunk_length, config.action_dim)
    assert torch.isfinite(actions).all()


def test_policy_prefix_conditioning_loss(model, batch):
    model.train()
    batch = dict(batch)
    batch["state_is_masked"] = torch.tensor([False], device=batch["state"].device)
    loss = model(
        batch,
        max_action_prefix=2,
        prefix_conditioning_prob=1.0,
        prefix_noise_scale=0.05,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    model.zero_grad(set_to_none=True)


def test_policy_rtc_sample_keeps_prefix(model, config, batch):
    model.eval()
    prefix = torch.randn(
        1,
        config.chunk_length,
        config.action_dim,
        device=batch["state"].device,
        dtype=batch["state"].dtype,
    )
    prefix_length = 2
    out = model.sample_actions_rtc(batch, prefix, prefix_length=prefix_length, num_steps=2)
    torch.testing.assert_close(out[:, :prefix_length], prefix[:, :prefix_length])
