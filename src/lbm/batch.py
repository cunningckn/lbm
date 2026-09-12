"""Convert dataloader batches into ``DiTPolicy`` inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lbm.dataloader.pad import embodiment_id_from_tag
from lbm.utils.preprocess import imagenet_normalize


def _one_policy_io(dataset) -> dict[str, Any]:
    io = getattr(dataset, "policy_io", None)
    if callable(io):
        io = io()
    if isinstance(io, dict) and "action_dim" in io and "camera_keys" in io:
        return {
            "camera_keys": tuple(io["camera_keys"]),
            "action_dim": int(io["action_dim"]),
            "state_dim": int(io.get("state_dim") or 0),
            "video_keys": tuple(io.get("video_keys") or ()),
            "chunk_length": int(io.get("chunk_length") or 1),
            "embodiment_id": int(io["embodiment_id"]) if io.get("embodiment_id") is not None else None,
        }
    video_keys = list(dataset.modality_keys.get("video", []))
    cameras = tuple(k.split(".", 1)[-1] for k in video_keys)
    modalities = dataset.metadata.modalities

    def _dim(keys: list[str], store) -> int:
        total = 0
        for key in keys:
            sub = key.split(".", 1)[-1]
            if sub not in store:
                raise KeyError(f"{key} not in metadata ({list(store)})")
            total += int(np.prod(store[sub].shape))
        return total

    action_keys = list(dataset.modality_keys.get("action", []))
    state_keys = list(dataset.modality_keys.get("state", []))
    chunk = 1
    deltas = getattr(dataset, "_delta_indices", None) or {}
    if action_keys and action_keys[0] in deltas:
        chunk = int(len(deltas[action_keys[0]]))
    tag = getattr(dataset, "tag", None)
    return {
        "camera_keys": cameras,
        "action_dim": _dim(action_keys, modalities.action),
        "state_dim": _dim(state_keys, modalities.state) if state_keys else 0,
        "video_keys": tuple(video_keys),
        "chunk_length": chunk,
        "embodiment_id": embodiment_id_from_tag(tag),
    }


def merge_policy_io(ios: list[dict[str, Any]]) -> dict[str, Any]:
    """Union cameras and take max dims / chunk length across a mixture."""
    if not ios:
        raise ValueError("merge_policy_io requires at least one IO spec")
    cameras: list[str] = []
    seen: set[str] = set()
    video_keys: list[str] = []
    for io in ios:
        for cam in io.get("camera_keys") or ():
            if cam not in seen:
                seen.add(cam)
                cameras.append(cam)
        for vk in io.get("video_keys") or ():
            if vk not in video_keys:
                video_keys.append(vk)
    return {
        "camera_keys": tuple(cameras),
        "action_dim": max(int(io.get("action_dim") or 0) for io in ios),
        "state_dim": max(int(io.get("state_dim") or 0) for io in ios),
        "video_keys": tuple(video_keys),
        "chunk_length": max(int(io.get("chunk_length") or 1) for io in ios),
    }


def infer_policy_io(dataset) -> dict[str, Any]:
    """Camera names and concatenated state/action dims from a LeRobot dataset or mixture."""
    datasets = list(dataset.datasets) if hasattr(dataset, "datasets") else [dataset]
    return merge_policy_io([_one_policy_io(ds) for ds in datasets])


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size:
            return _as_str(value.reshape(-1)[0])
        return ""
    if isinstance(value, (list, tuple)) and value:
        return _as_str(value[0])
    return str(value)


def _current_vector(x: torch.Tensor) -> torch.Tensor:
    """``(B, D)`` or ``(B, T, D)`` → current-step ``(B, D)``."""
    if x.ndim == 3:
        return x[:, -1]
    if x.ndim == 2:
        return x
    raise ValueError(f"expected state (B, D) or (B, T, D), got {tuple(x.shape)}")


def _normalize_images(
    image: torch.Tensor,
    camera_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """``(B, C, T, H, W, 3)`` uint8 → ImageNet CHW (last frame, or all T)."""
    if image.ndim != 6:
        raise ValueError(f"expected image (B, n_cams, T, H, W, 3), got {tuple(image.shape)}")
    n_cams = image.shape[1]
    if n_cams != len(camera_keys):
        raise ValueError(
            f"image has {n_cams} cameras, config has {len(camera_keys)}: {camera_keys}"
        )
    t_hist = image.shape[2]
    out: dict[str, torch.Tensor] = {}
    for i, cam in enumerate(camera_keys):
        frames = image[:, i].permute(0, 1, 4, 2, 3).float().div_(255.0)
        if t_hist == 1:
            out[cam] = imagenet_normalize(frames[:, -1].contiguous())
            continue
        bsz, t_dim, ch, h, w = frames.shape
        normed = imagenet_normalize(frames.reshape(bsz * t_dim, ch, h, w))
        out[cam] = normed.reshape(bsz, t_dim, ch, h, w)
    return out


def policy_batch_from_loader(
    batch: dict[str, Any],
    *,
    camera_keys: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
    embedder=None,
    language_encoder: str = "none",
    tokenize_fn=None,
    mask_state_ratio: float = 0.0,
    train: bool = True,
    non_blocking: bool = True,
    state_dim: int = 0,
    action_dim: int = 0,
    action_steps: int = 0,
) -> dict[str, Any]:
    """Map a collated LeRobot batch onto ``DiTPolicy.forward`` / ``sample_actions``."""
    image = batch["image"]
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    keys = tuple(batch.get("camera_keys") or camera_keys)
    if len(keys) != image.shape[1]:
        keys = camera_keys
    # Transfer compact uint8 frames before normalization; keep pinned DMA useful.
    image = image.to(device=device, non_blocking=non_blocking)
    images = _normalize_images(image, keys)
    images = {
        cam: img.to(device=device, dtype=dtype, non_blocking=non_blocking)
        for cam, img in images.items()
    }

    actions = torch.as_tensor(batch["action"]).to(
        device=device, dtype=dtype, non_blocking=non_blocking
    )
    if actions.ndim != 3:
        raise ValueError(f"expected action (B, T, D), got {tuple(actions.shape)}")

    if "state" in batch:
        state = _current_vector(
            torch.as_tensor(batch["state"]).to(
                device=device, dtype=dtype, non_blocking=non_blocking
            )
        )
    else:
        state = torch.zeros(
            actions.shape[0], int(state_dim), device=device, dtype=dtype
        )

    if state_dim:
        if state.shape[-1] > state_dim:
            raise ValueError("batch state exceeds model state_dim")
        state = F.pad(state, (0, state_dim - state.shape[-1]))
    original_action_shape = actions.shape
    if action_dim:
        if actions.shape[-1] > action_dim:
            raise ValueError("batch action exceeds model action_dim")
        actions = F.pad(actions, (0, action_dim - actions.shape[-1]))
    if action_steps:
        if actions.shape[1] > action_steps:
            raise ValueError("batch action exceeds model action_steps")
        actions = F.pad(actions, (0, 0, 0, action_steps - actions.shape[1]))

    langs = [_as_str(t) for t in batch["lang"]]
    bsz = actions.shape[0]
    if "embodiment_id" in batch:
        embodiment_id = torch.as_tensor(batch["embodiment_id"]).to(
            device=device, dtype=torch.long, non_blocking=non_blocking
        )
    else:
        tags = list(batch.get("robot_tag") or [])
        ids = [
            embodiment_id_from_tag(tags[i] if i < len(tags) else None) for i in range(bsz)
        ]
        embodiment_id = torch.as_tensor(ids, device=device, dtype=torch.long)

    out: dict[str, Any] = {
        "state": state,
        "actions": actions,
        "images": images,
        "lang": langs,
        "embodiment_id": embodiment_id,
        "robot_tag": list(batch.get("robot_tag") or []),
    }
    if "action_mask" in batch or actions.shape != original_action_shape:
        mask = torch.as_tensor(batch.get("action_mask", True), device=device, dtype=torch.bool)
        mask = torch.broadcast_to(mask, original_action_shape)
        out["action_mask"] = F.pad(
            mask, (0, actions.shape[-1] - original_action_shape[-1],
                   0, actions.shape[1] - original_action_shape[1]), value=False,
        )

    unknown = set(images) - set(camera_keys)
    if unknown:
        raise ValueError(f"batch cameras not present in model: {sorted(unknown)}")
    if "camera_mask" in batch or keys != camera_keys:
        source_mask = torch.as_tensor(batch.get("camera_mask", True), device=device, dtype=torch.bool)
        source_mask = torch.broadcast_to(source_mask, (bsz, len(keys)))
        camera_mask = torch.zeros(bsz, len(camera_keys), device=device, dtype=torch.bool)
        for i, key in enumerate(camera_keys):
            if key in keys:
                camera_mask[:, i] = source_mask[:, keys.index(key)]
        out["camera_mask"] = camera_mask
    template = next(iter(images.values()))
    out["images"] = {key: images[key] if key in images else torch.zeros_like(template) for key in camera_keys}

    if train and mask_state_ratio > 0 and state.numel():
        masked = torch.rand(state.shape[0], device=device) < mask_state_ratio
        out["state_is_masked"] = masked
        if masked.any():
            state = state.clone()
            state[masked] = 0
            out["state"] = state
    elif "state_is_masked" in batch:
        out["state_is_masked"] = torch.as_tensor(batch["state_is_masked"]).to(
            device=device, non_blocking=non_blocking
        )

    if tokenize_fn is not None:
        tokens, mask = tokenize_fn(langs)
        out["task_tokens"] = tokens.to(device=device, non_blocking=non_blocking)
        out["task_token_mask"] = mask.to(
            device=device, dtype=dtype, non_blocking=non_blocking
        )
    elif embedder is not None and language_encoder == "clip":
        tokens, mask = embedder.tokenize(langs)
        out["task_tokens"] = tokens.to(device=device, non_blocking=non_blocking)
        out["task_token_mask"] = mask.to(
            device=device, dtype=dtype, non_blocking=non_blocking
        )
    elif embedder is not None:
        out["task_vec_clip"] = embedder.encode(langs).to(
            device=device, dtype=dtype, non_blocking=non_blocking
        )
    return out
