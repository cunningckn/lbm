"""Synthetic batches for examples, benchmarks, and tests."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from lbm.config import DiTConfig

IMAGE_SIZE = 224


def make_fake_batch(
    config: DiTConfig,
    batch_size: int = 2,
    *,
    image_size: int = IMAGE_SIZE,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    include_actions: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Random tensors matching `DiTPolicy.forward` / `sample_actions` I/O.

    Images are already in the ImageNet-normalized CHW layout the model expects
    (`(B, 3, H, W)` per camera). `task_vec_clip` is a stand-in for CLIP features.
    """
    if image_size <= 0 or image_size % 16:
        raise ValueError(f"image_size must be a positive multiple of 16, got {image_size}")

    def rand(*shape):
        t = torch.randn(*shape, generator=generator, dtype=dtype)
        return t.to(device) if device is not None else t

    batch: dict[str, Any] = {
        "state": rand(batch_size, config.state_dim),
        "images": {cam: rand(batch_size, 3, image_size, image_size) for cam in config.camera_keys},
        "task_vec_clip": rand(batch_size, config.task_embed_dim),
        "task_tokens": torch.randint(
            1,
            max(2, config.t5_vocab_size if config.language_encoder == "t5" else 49408),
            (batch_size, config.language_max_length),
            generator=generator,
        ),
    }
    tokens = batch["task_tokens"]
    if device is not None:
        tokens = tokens.to(device)
    batch["task_tokens"] = tokens
    batch["task_token_mask"] = (tokens != 0).to(dtype=dtype)
    if include_actions:
        batch["actions"] = rand(batch_size, config.chunk_length, config.action_dim)
    return batch


def make_fake_sample(
    config: DiTConfig,
    *,
    image_size: int = IMAGE_SIZE,
    dtype: torch.dtype = torch.float32,
    include_actions: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Unbatched sample for `Dataset.__getitem__` / DataLoader collate."""
    batch = make_fake_batch(
        config,
        batch_size=1,
        image_size=image_size,
        dtype=dtype,
        include_actions=include_actions,
        generator=generator,
    )
    sample: dict[str, Any] = {
        "state": batch["state"][0],
        "images": {cam: img[0] for cam, img in batch["images"].items()},
        "task_vec_clip": batch["task_vec_clip"][0],
        "task_tokens": batch["task_tokens"][0],
        "task_token_mask": batch["task_token_mask"][0],
    }
    if include_actions:
        sample["actions"] = batch["actions"][0]
    return sample


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of `make_fake_sample` dicts into a model batch."""
    first = samples[0]
    out: dict[str, Any] = {
        "state": torch.stack([s["state"] for s in samples], dim=0),
        "images": {
            cam: torch.stack([s["images"][cam] for s in samples], dim=0) for cam in first["images"]
        },
        "task_vec_clip": torch.stack([s["task_vec_clip"] for s in samples], dim=0),
        "task_tokens": torch.stack([s["task_tokens"] for s in samples], dim=0),
        "task_token_mask": torch.stack([s["task_token_mask"] for s in samples], dim=0),
    }
    if "actions" in first:
        out["actions"] = torch.stack([s["actions"] for s in samples], dim=0)
    if "state_is_masked" in first:
        out["state_is_masked"] = torch.stack([s["state_is_masked"] for s in samples], dim=0)
    return out


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "images":
            out[key] = {cam: img.to(device, non_blocking=non_blocking) for cam, img in value.items()}
        elif torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=non_blocking)
        else:
            out[key] = value
    return out


def describe_batch(batch: dict[str, Any]) -> str:
    """Human-readable tensor spec for fake / real batches."""
    lines = []
    for key, value in batch.items():
        if key == "images":
            for cam, img in value.items():
                lines.append(f"images[{cam!r}]: {tuple(img.shape)} {img.dtype}")
        elif torch.is_tensor(value):
            lines.append(f"{key}: {tuple(value.shape)} {value.dtype}")
        else:
            lines.append(f"{key}: {type(value).__name__}")
    return "\n".join(lines)


class FakeActionDataset(Dataset):
    """On-the-fly synthetic trajectories. Regenerated each `__getitem__` (not cached)."""

    def __init__(
        self,
        config: DiTConfig,
        length: int = 256,
        *,
        image_size: int = IMAGE_SIZE,
        seed: int = 0,
        include_actions: bool = True,
    ):
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        self.config = config
        self.length = length
        self.image_size = image_size
        self.seed = seed
        self.include_actions = include_actions

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(self.seed + int(index))
        return make_fake_sample(
            self.config,
            image_size=self.image_size,
            include_actions=self.include_actions,
            generator=gen,
        )
