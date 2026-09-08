"""Attention helpers: prefer ``flash_attn`` on GPU bf16/fp16, else SDPA."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_FLASH_ATTN = None
_FLASH_ATTN_TRIED = False
_FORCE_FLASH: bool | None = None

# HIP: torch flash/mem SDPA faults; force math.
if getattr(torch.version, "hip", None):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def set_use_flash_attn(enabled: bool | None) -> None:
    """Force flash_attn on/off. ``None`` = auto."""
    global _FORCE_FLASH
    _FORCE_FLASH = enabled


def _flash_attn_func():
    global _FLASH_ATTN, _FLASH_ATTN_TRIED
    if not _FLASH_ATTN_TRIED:
        _FLASH_ATTN_TRIED = True
        try:
            from flash_attn import flash_attn_func as _fn

            _FLASH_ATTN = _fn
        except Exception:
            _FLASH_ATTN = None
    return _FLASH_ATTN


def configure_torch_sdp(*, mode: str = "auto") -> str:
    """Configure PyTorch SDPA backends (used by ``nn.MultiheadAttention``)."""
    mode = mode.lower().strip()
    if mode not in {"auto", "math", "flash"}:
        raise ValueError(f"unknown sdp mode: {mode}")
    env = os.environ.get("lbm_TORCH_SDP")
    if env:
        mode = env.lower().strip()
    if mode == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return "flash"
    if mode == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return "math"
    return "auto"


def split_qkv(qkv: torch.Tensor, num_heads: int, head_dim: int):
    """``(B, S, 3*H*D)`` → ``q, k, v`` each ``(B, H, S, D)``."""
    b, s, _ = qkv.shape
    qkv = qkv.reshape(b, s, 3, num_heads, head_dim)
    return qkv.permute(2, 0, 3, 1, 4).unbind(0)


def split_q(q: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    b, s, _ = q.shape
    return q.reshape(b, s, num_heads, head_dim).transpose(1, 2)


def split_kv(kv: torch.Tensor, num_heads: int, head_dim: int):
    """``(B, S, 2*H*D)`` → ``k, v`` each ``(B, H, S, D)``."""
    b, s, _ = kv.shape
    kv = kv.reshape(b, s, 2, num_heads, head_dim)
    return kv.permute(2, 0, 3, 1, 4).unbind(0)


def merge_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``(B, H, S, D)`` → ``(B, S, H*D)``."""
    b, _, s, _ = x.shape
    return x.transpose(1, 2).reshape(b, s, num_heads * head_dim)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    use_flash: bool | None = None,
) -> torch.Tensor:
    """Multi-head attention on ``q,k,v`` shaped ``(B, H, S, D)``."""
    if use_flash is None:
        use_flash = _FORCE_FLASH
    if use_flash is None:
        use_flash = (
            q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and _flash_attn_func() is not None
        )
    if use_flash:
        fn = _flash_attn_func()
        if fn is None:
            raise RuntimeError("flash_attn is not available")
        out = fn(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            causal=causal,
        )
        return out.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def attn_out(q, k, v, num_heads: int, head_dim: int, *, causal: bool = False) -> torch.Tensor:
    """Attention then merge heads to ``(B, S, H*D)``."""
    return merge_heads(attention(q, k, v, causal=causal), num_heads, head_dim)
