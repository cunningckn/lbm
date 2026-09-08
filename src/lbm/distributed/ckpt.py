"""Torch Distributed Checkpoint (DCP) save/load for Megatron-FSDP."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_fsdp_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.distributed.checkpoint.save(state, checkpoint_id=str(path))


def load_fsdp_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.distributed.checkpoint.load(state_dict=state, checkpoint_id=str(path))
    model.load_state_dict(state["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
