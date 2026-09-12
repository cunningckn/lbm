"""Validation lifecycle and distributed metric aggregation."""

from __future__ import annotations

import torch
import torch.distributed as dist

from lbm.training_metrics import action_error_stats


@torch.no_grad()
def evaluate_actions(model, module, loader, to_policy, *, device, num_steps: int, distributed: bool = False):
    """Return summed action error/count and restore the caller's mode on failure."""
    was_training = model.training
    try:
        model.eval()
        stats = torch.zeros(2, device=device, dtype=torch.float64)
        if loader is not None:
            for raw in loader:
                batch = to_policy(raw, train=False)
                pred = module.sample_actions(batch, num_steps=num_steps)
                stats += action_error_stats(pred, batch["actions"], batch.get("action_mask"))
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats
    finally:
        model.train(was_training)
