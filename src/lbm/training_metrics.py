"""Pure metric helpers shared by the training and validation loops."""

from __future__ import annotations

import torch


def action_error_stats(pred, actions, action_mask=None) -> torch.Tensor:
    """Return squared-error sum and valid element count in float64-safe form."""
    error = (pred.float() - actions.float()).square()
    if action_mask is None:
        count = error.new_tensor(error.numel())
    else:
        valid = torch.broadcast_to(action_mask.to(device=error.device, dtype=torch.bool), error.shape)
        error = error.masked_fill(~valid, 0.0)
        count = valid.sum().to(dtype=error.dtype)
    return torch.stack((error.sum(), count))
