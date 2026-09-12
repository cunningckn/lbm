"""Validation lifecycle and distributed metric aggregation."""

from __future__ import annotations

import torch
import torch.distributed as dist

from lbm.training_metrics import action_error_stats


@torch.no_grad()
def evaluate_actions(model, module, loader, to_policy, *, device, num_steps: int, distributed: bool = False):
    """Return summed action error/count and restore the caller's mode on failure."""
    if isinstance(loader, dict):
        total = torch.zeros(2, device=device, dtype=torch.float64)
        for name, source_loader in loader.items():
            stats = evaluate_actions(model, module, source_loader, to_policy, device=device,
                                     num_steps=num_steps, distributed=distributed)
            total += stats
            if (not distributed or dist.get_rank() == 0) and stats[1].item():
                print(f"validation source={name} recon_error={(stats[0] / stats[1]).item():.6f}")
        return total
    was_training = model.training
    try:
        model.eval()
        stats = torch.zeros(2, device=device, dtype=torch.float64)
        if loader is not None:
            seen = 0
            for raw in loader:
                batch = to_policy(raw, train=False)
                pred = (module.sample_actions(batch, num_steps=num_steps)
                        if hasattr(module, "sample_actions") else model(batch, sample_steps=num_steps))
                mask = batch.get("action_mask")
                valid_count = getattr(getattr(loader, 'dataset', None), 'valid_count', None)
                if valid_count is not None:
                    if mask is None:
                        mask = torch.ones_like(batch['actions'], dtype=torch.bool)
                    valid_rows = torch.arange(len(pred), device=pred.device) + seen < valid_count
                    mask = mask.bool() & valid_rows[:, None, None]
                seen += len(pred)
                stats += action_error_stats(pred, batch["actions"], mask)
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats
    finally:
        model.train(was_training)
