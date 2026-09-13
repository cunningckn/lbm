"""Relative-time features and optional residual state-history attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def history_contract(mask, offsets, *, batch, steps, device):
    if mask is None or offsets is None:
        raise ValueError("timed history requires validity and time offsets")
    mask = mask.to(device=device, dtype=torch.bool)
    offsets = offsets.to(device=device, dtype=torch.float32)
    if mask.shape != (batch, steps) or offsets.shape != mask.shape:
        raise ValueError("history validity/offsets must match (batch, history)")
    if not torch.isfinite(offsets).all() or (offsets > 1e-6).any():
        raise ValueError("history offsets must be finite and nonpositive")
    return mask, offsets


def time_features(offsets, width):
    """Parameter-free relative time; exactly zero for the current observation."""
    half = (width + 1) // 2
    frequency = torch.exp(
        torch.arange(half, device=offsets.device, dtype=torch.float32) * (-math.log(10000) / max(half - 1, 1))
    )
    phase = offsets.float()[..., None] * frequency
    return torch.cat((phase.sin(), phase.cos() - 1), dim=-1)[..., :width]


class StateHistoryPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, current, history, valid, offsets):
        history = history.masked_fill(~valid[..., None], 0)
        timed = history + time_features(offsets, history.shape[-1]).to(history.dtype)
        query = self.query(current)[:, None, None, :]
        key = self.key(timed)[:, None]
        value = self.value(history)[:, None]
        pooled = F.scaled_dot_product_attention(query, key, value, attn_mask=valid[:, None, None, :])[:, 0, 0]
        return current + self.gate.tanh() * pooled
