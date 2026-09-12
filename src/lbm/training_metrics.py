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


def log_train_metrics(*, step, loss, lr, grad_norm, steps_per_s, logger=None):
    """Keep console and experiment-logger fields together."""
    loss = float(loss)
    grad_norm = float(grad_norm)
    print(f"step {step:6d}  loss {loss:.4f}  lr {lr:.2e}  gnorm {grad_norm:.3f}  {steps_per_s:.2f} it/s")
    if logger:
        logger.log({"loss": loss, "lr": lr, "grad_norm": grad_norm, "steps_per_s": steps_per_s}, step=step)


def log_validation_metrics(stats, *, step, logger=None):
    if stats[1].item() > 0:
        recon = (stats[0] / stats[1]).item()
        print(f"step {step:6d}  val_recon_error {recon:.4f}")
        if logger:
            logger.log({"val_recon_error": recon}, step=step)
    else:
        print(f"step {step:6d}  val skipped (no valid action elements in full validation batches)")
