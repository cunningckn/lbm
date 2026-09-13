"""Atomic training checkpoints with explicit replay state.

Only load trusted checkpoints: optimizer and NumPy RNG state use pickle.
"""

from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 2


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, *, step, epoch, batch_in_epoch, epoch_rng, signature):
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "epoch_rng": epoch_rng,
        "rng_state": capture_rng_state(),
        "signature": signature,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as out:
            torch.save(payload, out)
            out.flush()
            os.fsync(out.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer, scheduler, *, signature):
    """Validate replay compatibility before applying any state; restore RNG later."""
    # Modern path-based checkpoints can stay file-backed until used, avoiding a
    # full anonymous CPU copy before DataLoader workers fork. Private mappings
    # keep CPU optimizer updates from changing the saved checkpoint on disk.
    mapped = os.name == "posix" and isinstance(path, (str, os.PathLike)) and zipfile.is_zipfile(path)
    if mapped:
        with torch.serialization.set_default_mmap_options(torch.serialization.MAP_PRIVATE):
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    else:
        # Preserve legacy serialization and seekable stream support.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model", "optimizer", "scheduler", "step", "epoch", "batch_in_epoch", "epoch_rng", "rng_state", "signature",
    }
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION or required - payload.keys():
        raise ValueError("Checkpoint lacks complete replay state; use --ckpt for weight-only initialization")
    if payload["signature"] != signature:
        raise ValueError("Resume configuration or dataset length differs from checkpoint; use --ckpt for a new run")
    step, epoch, cursor = (payload[key] for key in ("step", "epoch", "batch_in_epoch"))
    batches = signature["batches_per_epoch"]
    if (
        any(type(value) is not int or value < 0 for value in (step, epoch, cursor))
        or not 0 <= cursor <= batches
        or step != epoch * batches + cursor
    ):
        raise ValueError("Invalid checkpoint step/epoch/batch cursor")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return payload


def standard_checkpoint_paths(output_dir: str | Path, step: int) -> tuple[Path, Path]:
    """Return the numbered and rolling paths for a standard training checkpoint."""
    root = Path(output_dir)
    return root / f"{step}.pt", root / "last.pt"


def save_standard_checkpoint(output_dir: str | Path, model, optimizer, scheduler, *, step: int,
                             epoch: int, batch_in_epoch: int, epoch_rng, signature) -> Path:
    """Atomically save both the numbered and rolling standard checkpoints."""
    numbered, rolling = standard_checkpoint_paths(output_dir, step)
    for path in (numbered, rolling):
        save_checkpoint(path, model, optimizer, scheduler, step=step, epoch=epoch,
                        batch_in_epoch=batch_in_epoch, epoch_rng=epoch_rng, signature=signature)
    return numbered
