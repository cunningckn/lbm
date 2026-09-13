"""Compact prefix modulation, with optional compilation of pointwise operations."""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch


class PrefixConditioning(NamedTuple):
    normal: torch.Tensor
    prefix: torch.Tensor
    mask: torch.Tensor
    width: int


class TokenCondition(NamedTuple):
    head: torch.Tensor
    tail: torch.Tensor


def modulate(x, shift, scale):
    if isinstance(shift, TokenCondition):
        width = shift.head.shape[1]
        head, tail = x.split((width, x.shape[1] - width), dim=1)
        return torch.cat((head * (1 + scale.head) + shift.head,
                          tail * (1 + scale.tail) + shift.tail), dim=1)
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def gate_residual(gate, residual):
    if isinstance(gate, TokenCondition):
        width = gate.head.shape[1]
        head, tail = residual.split((width, residual.shape[1] - width), dim=1)
        return torch.cat((gate.head * head, gate.tail * tail), dim=1)
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return gate * residual


def conditioning_chunks(layer, conditioning, count):
    """Project repeated prefix/non-prefix conditions once per sample."""
    if isinstance(conditioning, PrefixConditioning):
        normal, prefix, mask, width = conditioning
        normal = layer(normal).chunk(count, dim=-1)
        prefix = layer(prefix).chunk(count, dim=-1)
        return tuple(TokenCondition(torch.where(mask[:, :width, None], p[:, None], n[:, None]), n[:, None])
                     for n, p in zip(normal, prefix))
    if isinstance(conditioning, tuple):
        normal, prefix, mask = conditioning
        normal = layer(normal).chunk(count, dim=-1)
        prefix = layer(prefix).chunk(count, dim=-1)
        return tuple(torch.where(mask[..., None], p[:, None], n[:, None])
                     for n, p in zip(normal, prefix))
    return layer(conditioning).chunk(count, dim=-1)


@lru_cache(maxsize=1)
def compiled_conditioning_functions():
    # These small graphs need no parallel compilation pool. Bound compilation
    # memory rather than starting one compiler worker per host CPU.
    import torch._inductor.config as inductor_config

    inductor_config.compile_threads = 1
    # Preserve eager BF16 rounding at intermediate pointwise operations.
    options = {"emulate_precision_casts": True}
    return (torch.compile(modulate, fullgraph=True, options=options),
            torch.compile(gate_residual, fullgraph=True, options=options))
