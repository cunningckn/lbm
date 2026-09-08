"""Shared building blocks for DiT and encoder towers."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class FeedForward(nn.Module):
    """Two-layer MLP: Linear → GELU → Linear."""

    def __init__(self, dim: int, hidden: int, *, approximate: str = "none"):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU(approximate=approximate)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTMlp(FeedForward):
    """Dense DiT FFN (GELU-tanh)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__(dim, hidden, approximate="tanh")


class BFloat16Mixin:
    """Optional CUDA bf16 autocast; outputs are cast back to fp32."""

    bfloat16 = False

    def set_bfloat16(self, enabled: bool = True) -> None:
        self.bfloat16 = bool(enabled)

    def _maybe_bf16(self, probe: torch.Tensor, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        if self.bfloat16 and probe.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return fn().to(torch.float32)
        return fn()


def run_maybe_frozen(trainable: bool, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    if trainable:
        return fn()
    with torch.no_grad():
        return fn()
