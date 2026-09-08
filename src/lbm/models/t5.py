"""T5 encoder (t5-small defaults) for language conditioning."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lbm.config import DiTConfig, default_checkpoints_dir
from lbm.models.common import BFloat16Mixin
from lbm.models.weights import (
    ensure_hf_cached,
    env_paths,
    first_existing,
    layer_ids,
    load_into,
    local_checkpoint_candidates,
    read_checkpoint,
    strip_prefixes,
)

T5_HF_REPO = "google-t5/t5-small"
T5_HF_FILE = "model.safetensors"
_HF_PREFIXES = ("model.",)


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(orig_dtype)


def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    num_buckets //= 2
    relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = (
        max_exact
        + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
    )
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )
    return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)


def _additive_pad_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """HF T5 inverted mask: 0 keep, ``finfo(dtype).min`` pad, shape ``(B, 1, 1, T)``."""
    inverted = 1.0 - attention_mask.to(dtype=dtype)
    return inverted[:, None, None, :] * torch.finfo(dtype).min


class T5Attention(nn.Module):
    def __init__(self, d_model, num_heads, has_relative_bias: bool, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        inner = num_heads * self.head_dim
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)
        self.has_relative_bias = has_relative_bias
        if has_relative_bias:
            self.relative_attention_bias = nn.Embedding(32, num_heads)

    def _position_bias(self, length, device):
        context = torch.arange(length, device=device)[:, None]
        memory = torch.arange(length, device=device)[None, :]
        relative = _relative_position_bucket(memory - context)
        bias = self.relative_attention_bias(relative)
        return bias.permute(2, 0, 1).unsqueeze(0)

    def forward(self, x, mask, position_bias):
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        if position_bias is None and self.has_relative_bias:
            position_bias = self._position_bias(n, x.device)
        if position_bias is not None:
            scores = scores + position_bias
        if mask is not None:
            scores = scores + mask.to(dtype=scores.dtype)
        attn = torch.softmax(scores.float(), dim=-1).type_as(scores)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.o(out), position_bias


class T5Layer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, has_relative_bias: bool, dropout: float = 0.1):
        super().__init__()
        self.self_attn = T5Attention(d_model, num_heads, has_relative_bias, dropout=dropout)
        self.norm1 = T5LayerNorm(d_model)
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.norm2 = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, position_bias):
        y, position_bias = self.self_attn(self.norm1(x), mask, position_bias)
        x = x + self.dropout(y)
        h = self.dropout(F.relu(self.wi(self.norm2(x))))
        x = x + self.dropout(self.wo(h))
        return x, position_bias


class T5LanguageEncoder(BFloat16Mixin, nn.Module):
    """Original T5 encoder hidden states ``(B, T, d_model)``. Pooling is ``LanguagePoolProj``."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        d_model = config.t5_d_model
        self.out_dim = d_model
        self.bfloat16 = False
        dropout = float(config.t5_dropout)
        self.embed_tokens = nn.Embedding(config.t5_vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                T5Layer(
                    d_model,
                    config.t5_d_ff,
                    config.t5_num_heads,
                    has_relative_bias=(i == 0),
                    dropout=dropout,
                )
                for i in range(config.t5_num_layers)
            ]
        )
        self.final_norm = T5LayerNorm(d_model)
        nn.init.normal_(self.embed_tokens.weight, std=1.0)

    def forward(self, input_ids, attention_mask=None):
        return self._maybe_bf16(input_ids, lambda: self._encode(input_ids, attention_mask))

    def _encode(self, input_ids, attention_mask):
        x = self.dropout(self.embed_tokens(input_ids))
        if attention_mask is None:
            attention_mask = (input_ids != 0).to(dtype=x.dtype)
        else:
            attention_mask = attention_mask.to(dtype=x.dtype)
        mask = _additive_pad_mask(attention_mask, x.dtype)
        position_bias = None
        for layer in self.layers:
            x, position_bias = layer(x, mask, position_bias)
        return self.dropout(self.final_norm(x))


def resolve_t5_path() -> Path | None:
    """First existing t5-small checkpoint among env vars and ``lbm/checkpoints/t5``."""
    return first_existing(
        *env_paths("LBM_T5", "lbm_T5"),
        *local_checkpoint_candidates(
            "t5",
            "t5-small.safetensors",
            "model.safetensors",
            "pytorch_model.bin",
        ),
    )


def ensure_t5_weights() -> Path:
    """Return a local t5-small checkpoint, downloading google-t5/t5-small if needed."""
    return ensure_hf_cached(
        resolve_t5_path,
        default_checkpoints_dir() / "t5" / "t5-small.safetensors",
        repo=T5_HF_REPO,
        filename=T5_HF_FILE,
        url_env="lbm_T5_URL",
    )


def _is_hf_t5(state: dict[str, torch.Tensor]) -> bool:
    return any("encoder.block." in k and "SelfAttention" in k for k in state)


def convert_hf_t5(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HuggingFace T5 encoder keys onto ``T5LanguageEncoder``."""
    state = strip_prefixes(state, _HF_PREFIXES)
    embed = state.get("encoder.embed_tokens.weight", state.get("shared.weight"))
    if embed is None:
        raise RuntimeError("T5 checkpoint missing shared / encoder.embed_tokens weights")
    out: dict[str, torch.Tensor] = {
        "embed_tokens.weight": embed,
        "final_norm.weight": state["encoder.final_layer_norm.weight"],
    }
    for i in layer_ids(state, "encoder.block.", 2):
        src = f"encoder.block.{i}"
        dst = f"layers.{i}"
        out[f"{dst}.self_attn.q.weight"] = state[f"{src}.layer.0.SelfAttention.q.weight"]
        out[f"{dst}.self_attn.k.weight"] = state[f"{src}.layer.0.SelfAttention.k.weight"]
        out[f"{dst}.self_attn.v.weight"] = state[f"{src}.layer.0.SelfAttention.v.weight"]
        out[f"{dst}.self_attn.o.weight"] = state[f"{src}.layer.0.SelfAttention.o.weight"]
        rel = f"{src}.layer.0.SelfAttention.relative_attention_bias.weight"
        if rel in state:
            out[f"{dst}.self_attn.relative_attention_bias.weight"] = state[rel]
        out[f"{dst}.norm1.weight"] = state[f"{src}.layer.0.layer_norm.weight"]
        out[f"{dst}.wi.weight"] = state[f"{src}.layer.1.DenseReluDense.wi.weight"]
        out[f"{dst}.wo.weight"] = state[f"{src}.layer.1.DenseReluDense.wo.weight"]
        out[f"{dst}.norm2.weight"] = state[f"{src}.layer.1.layer_norm.weight"]
    return out


def load_t5(module: T5LanguageEncoder, ckpt_path):
    """Load T5 encoder weights (native ``.pt`` or HuggingFace ``.safetensors`` / ``.bin``)."""
    sd = read_checkpoint(Path(ckpt_path).expanduser())
    if _is_hf_t5(sd):
        sd = convert_hf_t5(sd)
    return load_into(module, sd, name="T5")
