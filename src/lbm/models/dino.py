"""DINOv3 ViT-B/16 vision backbone."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lbm.config import DiTConfig
from lbm.models.attention import attn_out, split_qkv
from lbm.models.common import BFloat16Mixin, FeedForward
from lbm.models.weights import (
    cat_keys,
    env_paths,
    first_existing,
    layer_ids,
    load_into,
    local_checkpoint_candidates,
    read_checkpoint,
    strip_prefixes,
)


def _rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, rope, n_tokens):
    sin, cos = rope
    n_prefix = n_tokens - sin.shape[-2]  # cls + storage tokens are not rotated
    q_dt, k_dt = q.dtype, k.dtype
    q, k = q.to(sin.dtype), k.to(sin.dtype)

    def _rotate(x):
        prefix, rest = x[:, :, :n_prefix], x[:, :, n_prefix:]
        return torch.cat([prefix, rest * cos + _rope_rotate_half(rest) * sin], dim=-2)

    return _rotate(q).to(q_dt), _rotate(k).to(k_dt)


class DinoRope(nn.Module):
    """RoPE over the 2D patch grid (base=100, separate coord normalization).

    rescale_coords=2 applies a random log-uniform rescale of the coordinates
    during training only — part of the pretraining distribution, kept for
    finetuning fidelity.
    """

    def __init__(self, embed_dim, num_heads, base=100.0, rescale_coords=2.0):
        super().__init__()
        d_head = embed_dim // num_heads
        self.d_head = d_head
        self.rescale_coords = rescale_coords
        self.register_buffer("periods", torch.empty(d_head // 4), persistent=True)
        with torch.no_grad():
            self.periods.copy_(
                base ** (2 * torch.arange(d_head // 4, dtype=torch.float32) / (d_head // 2))
            )

    def forward(self, H, W):
        dev = self.periods.device
        coords_h = torch.arange(0.5, H, device=dev, dtype=torch.float32) / H
        coords_w = torch.arange(0.5, W, device=dev, dtype=torch.float32) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0
        if self.training and self.rescale_coords is not None:
            r = np.log(self.rescale_coords)
            rescale = torch.empty(1, device=dev).uniform_(-r, r).exp()
            coords = coords * rescale
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


class LinearKMaskedBias(nn.Linear):
    """qkv Linear whose k-third of the bias is masked to zero (DINOv3 quirk)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("bias_mask", torch.full_like(self.bias, math.nan))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias * self.bias_mask.to(self.bias.dtype))


class DinoAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = LinearKMaskedBias(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, rope=None):
        q, k, v = split_qkv(self.qkv(x), self.num_heads, self.head_dim)
        if rope is not None:
            q, k = _apply_rope(q, k, rope, x.shape[1])
        return self.proj(attn_out(q, k, v, self.num_heads, self.head_dim))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class DinoMlp(FeedForward):
    """GELU MLP used inside ``DinoBlock``."""


class DinoBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = DinoAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = DinoMlp(dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoPatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/16, W/16)
        return x.flatten(2).transpose(1, 2), x.shape[2], x.shape[3]


class DinoVisionTransformer(nn.Module):
    """DINOv3 ViT-B/16 with 4 storage tokens. encode_image_tokens() returns
    (B, 1+196, 768) = CLS + patch tokens (storage tokens dropped)."""

    N_STORAGE_TOKENS = 4

    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = DinoPatchEmbed(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.empty(1, self.N_STORAGE_TOKENS, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.rope_embed = DinoRope(embed_dim, num_heads)
        self.blocks = nn.ModuleList(
            DinoBlock(embed_dim, num_heads) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.init_weights()

    def init_weights(self):
        """Fill ``bias_mask`` (otherwise NaN) so the K-third of every qkv bias
        is masked to 0 — without this, a fresh DINOv3 produces NaN on its first
        forward pass."""
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if isinstance(m, LinearKMaskedBias):
                    o = m.out_features
                    m.bias_mask.fill_(1)
                    m.bias_mask[o // 3 : 2 * o // 3].fill_(0)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, LayerScale):
                nn.init.constant_(m.gamma, 1e-5)
            elif isinstance(m, DinoPatchEmbed):
                m.proj.reset_parameters()

    def encode_image_tokens(self, images):
        x, H, W = self.patch_embed(images)
        B = x.shape[0]
        cls_token = self.cls_token + 0 * self.mask_token
        x = torch.cat(
            [cls_token.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1), x], dim=1
        )
        rope = self.rope_embed(H, W)
        for blk in self.blocks:
            x = blk(x, rope=rope)
        x = self.norm(x)
        cls_out = x[:, :1]
        patches = x[:, 1 + self.N_STORAGE_TOKENS :]
        return torch.cat([cls_out, patches], dim=1)


class DinoVisionBackbone(BFloat16Mixin, nn.Module):
    """DINOv3 wrapper. Checkpoint keys stay under ``img_backbone.dinov3_model.*``."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.dinov3_model = DinoVisionTransformer(
            embed_dim=config.vit_embed_dim,
            depth=config.vit_depth,
            num_heads=config.vit_num_heads,
        )
        self.bfloat16 = False

    def encode_image_tokens(self, images):
        return self._maybe_bf16(images, lambda: self.dinov3_model.encode_image_tokens(images))


_HF_PREFIXES = ("model.", "backbone.", "dinov3_model.")
_NATIVE_SKIP = ("bias_mask", "rope_embed.periods")


def resolve_dinov3_path() -> Path | None:
    """First existing DINOv3 checkpoint among env vars and ``lbm/checkpoints/dinov3``."""
    return first_existing(
        *env_paths("LBM_DINO", "lbm_DINO", "DINOV3_CKPT"),
        *local_checkpoint_candidates(
            "dinov3",
            "dinov3_vitb16_pretrain_lvd1689m.pth",
            "model.safetensors",
        ),
    )


def ensure_dinov3_weights() -> Path:
    """Return a local DINOv3 checkpoint. DINOv3 is not auto-downloaded (license)."""
    path = resolve_dinov3_path()
    if path is None:
        raise FileNotFoundError(
            "DINOv3 checkpoint not found. Set lbm_DINO or place "
            "model.safetensors / dinov3_vitb16_pretrain_lvd1689m.pth in lbm/checkpoints/dinov3."
        )
    return path


def _is_hf_dinov3(state: dict[str, torch.Tensor]) -> bool:
    return any(k.startswith(("embeddings.", "layer.")) for k in state)


def convert_hf_dinov3(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HuggingFace DINOv3 keys onto ``DinoVisionTransformer``."""
    state = strip_prefixes(state, _HF_PREFIXES)
    out: dict[str, torch.Tensor] = {
        "cls_token": state["embeddings.cls_token"],
        "storage_tokens": state["embeddings.register_tokens"],
        "mask_token": state["embeddings.mask_token"].reshape(1, -1),
        "patch_embed.proj.weight": state["embeddings.patch_embeddings.weight"],
        "patch_embed.proj.bias": state["embeddings.patch_embeddings.bias"],
        "norm.weight": state["norm.weight"],
        "norm.bias": state["norm.bias"],
    }
    for i in layer_ids(state, "layer.", 1):
        src, dst = f"layer.{i}", f"blocks.{i}"
        out[f"{dst}.attn.qkv.weight"] = cat_keys(
            state,
            f"{src}.attention.q_proj.weight",
            f"{src}.attention.k_proj.weight",
            f"{src}.attention.v_proj.weight",
        )
        q_b = state[f"{src}.attention.q_proj.bias"]
        # HF k_proj has no bias (DINOv3: K bias is always 0). Pad zeros for fused qkv.
        k_b = state.get(f"{src}.attention.k_proj.bias", torch.zeros_like(q_b))
        out[f"{dst}.attn.qkv.bias"] = torch.cat(
            [q_b, k_b, state[f"{src}.attention.v_proj.bias"]], dim=0
        )
        out[f"{dst}.attn.proj.weight"] = state[f"{src}.attention.o_proj.weight"]
        out[f"{dst}.attn.proj.bias"] = state[f"{src}.attention.o_proj.bias"]
        out[f"{dst}.ls1.gamma"] = state[f"{src}.layer_scale1.lambda1"]
        out[f"{dst}.ls2.gamma"] = state[f"{src}.layer_scale2.lambda1"]
        out[f"{dst}.norm1.weight"] = state[f"{src}.norm1.weight"]
        out[f"{dst}.norm1.bias"] = state[f"{src}.norm1.bias"]
        out[f"{dst}.norm2.weight"] = state[f"{src}.norm2.weight"]
        out[f"{dst}.norm2.bias"] = state[f"{src}.norm2.bias"]
        out[f"{dst}.mlp.fc1.weight"] = state[f"{src}.mlp.up_proj.weight"]
        out[f"{dst}.mlp.fc1.bias"] = state[f"{src}.mlp.up_proj.bias"]
        out[f"{dst}.mlp.fc2.weight"] = state[f"{src}.mlp.down_proj.weight"]
        out[f"{dst}.mlp.fc2.bias"] = state[f"{src}.mlp.down_proj.bias"]
    return out


def load_dinov3(module: DinoVisionBackbone | DinoVisionTransformer, ckpt_path):
    """Load a DINOv3 checkpoint (native ``.pth`` or HuggingFace ``.safetensors``)."""
    sd = read_checkpoint(Path(ckpt_path).expanduser())
    if _is_hf_dinov3(sd):
        sd = convert_hf_dinov3(sd)
    target = module.dinov3_model if isinstance(module, DinoVisionBackbone) else module
    return load_into(target, sd, name="DINOv3", skip_missing=_NATIVE_SKIP)

