"""AdamW param groups that honor frozen vision modules and vision_lr_scale."""

from __future__ import annotations

import torch
from torch import nn

from lbm.config import OptimConfig


def _unwrap(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def build_adamw(model: nn.Module, optim: OptimConfig, *, fused: bool | None = None) -> torch.optim.AdamW:
    """Build AdamW, skipping frozen params. Vision encoder gets ``vision_lr_scale``."""
    root = _unwrap(model)
    backbone = getattr(root, "img_backbone", None)
    vision_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    vis = [p for p in (backbone.parameters() if backbone is not None else ()) if p.requires_grad]
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in vision_ids]
    groups = []
    if vis:
        groups.append({"params": vis, "lr": optim.learning_rate * optim.vision_lr_scale})
    if rest:
        groups.append({"params": rest, "lr": optim.learning_rate})
    if not groups:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(
        groups,
        lr=optim.learning_rate,
        betas=(optim.adam_beta1, optim.adam_beta2),
        eps=optim.adam_epsilon,
        weight_decay=optim.weight_decay,
        fused=fused,
    )


def count_trainable(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total
