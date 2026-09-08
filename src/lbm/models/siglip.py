"""SigLIP ViT-B/16 vision backbone (patch tokens, no class token)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from lbm.config import DiTConfig, default_checkpoints_dir
from lbm.models.attention import attn_out, split_qkv
from lbm.models.common import BFloat16Mixin, FeedForward
from lbm.models.weights import (
    cat_keys,
    ensure_hf_cached,
    env_paths,
    first_existing,
    layer_ids,
    load_into,
    local_checkpoint_candidates,
    read_checkpoint,
    strip_prefixes,
)

SIGLIP_HF_REPO = "google/siglip-base-patch16-224"
SIGLIP_HF_FILE = "model.safetensors"
_HF_PREFIXES = ("model.", "vision_model.", "siglip_model.", "vision_tower.")


class SiglipAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        q, k, v = split_qkv(self.qkv(x), self.num_heads, self.head_dim)
        return self.proj(attn_out(q, k, v, self.num_heads, self.head_dim))


class SiglipMlp(FeedForward):
    def __init__(self, dim, hidden):
        super().__init__(dim, hidden, approximate="tanh")


class SiglipBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SiglipAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SiglipMlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SiglipVisionTransformer(nn.Module):
    """SigLIP-B/16-style ViT: patch tokens only, learned absolute positions."""

    def __init__(self, embed_dim, depth, num_heads, image_size=224, patch_size=16, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        grid = image_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, embed_dim))
        self.blocks = nn.ModuleList(
            SiglipBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_image_tokens(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        n = x.shape[1]
        pos = self.pos_embed
        if pos.shape[1] != n:
            g0 = int(pos.shape[1] ** 0.5)
            g1 = int(n ** 0.5)
            pos = (
                nn.functional.interpolate(
                    pos.reshape(1, g0, g0, -1).permute(0, 3, 1, 2),
                    size=(g1, g1),
                    mode="bicubic",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .reshape(1, n, -1)
            )
        x = x + pos.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SiglipVisionBackbone(BFloat16Mixin, nn.Module):
    """SigLIP ViT-B/16 wrapper with the same encode_image_tokens contract as DINO."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.siglip_model = SiglipVisionTransformer(
            embed_dim=config.vit_embed_dim,
            depth=config.vit_depth,
            num_heads=config.vit_num_heads,
        )
        self.bfloat16 = False

    def encode_image_tokens(self, images):
        return self._maybe_bf16(images, lambda: self.siglip_model.encode_image_tokens(images))


def resolve_siglip_path() -> Path | None:
    """First existing SigLIP-B checkpoint among env vars and ``lbm/checkpoints/siglip``."""
    return first_existing(
        *env_paths("LBM_SIGLIP", "lbm_SIGLIP"),
        *local_checkpoint_candidates(
            "siglip",
            "siglip-base-patch16-224.safetensors",
            "model.safetensors",
        ),
    )


def ensure_siglip_weights() -> Path:
    """Return a local SigLIP-B checkpoint, downloading google/siglip-base-patch16-224 if needed."""
    return ensure_hf_cached(
        resolve_siglip_path,
        default_checkpoints_dir() / "siglip" / "siglip-base-patch16-224.safetensors",
        repo=SIGLIP_HF_REPO,
        filename=SIGLIP_HF_FILE,
        url_env="lbm_SIGLIP_URL",
    )


def _is_hf_siglip(state: dict[str, torch.Tensor]) -> bool:
    return any(k.endswith("embeddings.patch_embedding.weight") for k in state)


def convert_hf_siglip(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HuggingFace SigLIP vision keys onto ``SiglipVisionTransformer``."""
    state = strip_prefixes(state, _HF_PREFIXES)
    pos = state["embeddings.position_embedding.weight"]
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)
    out: dict[str, torch.Tensor] = {
        "patch_embed.weight": state["embeddings.patch_embedding.weight"],
        "patch_embed.bias": state["embeddings.patch_embedding.bias"],
        "pos_embed": pos,
        "norm.weight": state["post_layernorm.weight"],
        "norm.bias": state["post_layernorm.bias"],
    }
    for i in layer_ids(state, "encoder.layers.", 2):
        src, dst = f"encoder.layers.{i}", f"blocks.{i}"
        out[f"{dst}.attn.qkv.weight"] = cat_keys(
            state,
            f"{src}.self_attn.q_proj.weight",
            f"{src}.self_attn.k_proj.weight",
            f"{src}.self_attn.v_proj.weight",
        )
        out[f"{dst}.attn.qkv.bias"] = cat_keys(
            state,
            f"{src}.self_attn.q_proj.bias",
            f"{src}.self_attn.k_proj.bias",
            f"{src}.self_attn.v_proj.bias",
        )
        out[f"{dst}.attn.proj.weight"] = state[f"{src}.self_attn.out_proj.weight"]
        out[f"{dst}.attn.proj.bias"] = state[f"{src}.self_attn.out_proj.bias"]
        out[f"{dst}.norm1.weight"] = state[f"{src}.layer_norm1.weight"]
        out[f"{dst}.norm1.bias"] = state[f"{src}.layer_norm1.bias"]
        out[f"{dst}.norm2.weight"] = state[f"{src}.layer_norm2.weight"]
        out[f"{dst}.norm2.bias"] = state[f"{src}.layer_norm2.bias"]
        out[f"{dst}.mlp.fc1.weight"] = state[f"{src}.mlp.fc1.weight"]
        out[f"{dst}.mlp.fc1.bias"] = state[f"{src}.mlp.fc1.bias"]
        out[f"{dst}.mlp.fc2.weight"] = state[f"{src}.mlp.fc2.weight"]
        out[f"{dst}.mlp.fc2.bias"] = state[f"{src}.mlp.fc2.bias"]
    return out


def load_siglip(module: SiglipVisionBackbone | SiglipVisionTransformer, ckpt_path):
    """Load SigLIP vision weights (native ``.pt`` or HuggingFace ``.safetensors``)."""
    sd = read_checkpoint(Path(ckpt_path).expanduser())
    sd = convert_hf_siglip(sd) if _is_hf_siglip(sd) else strip_prefixes(sd, ("siglip_model.", "model."))
    target = module.siglip_model if isinstance(module, SiglipVisionBackbone) else module
    return load_into(target, sd, name="SigLIP")
