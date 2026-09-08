"""Pad heterogeneous samples so a mixture batch can stack."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lbm.dataloader.embodiment import DEFAULT_EMBODIMENT_ID, embodiment_id_from_tag

__all__ = [
    "DEFAULT_EMBODIMENT_ID",
    "attach_batch_ids_and_masks",
    "collate_fn",
    "dataloader_worker_init_fn",
    "embodiment_id_from_tag",
    "pad_loader_batch",
]


def _as_thwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        return arr[None]
    if arr.ndim != 4:
        raise ValueError(f"expected image (H,W,3) or (T,H,W,3), got {arr.shape}")
    return arr


def _as_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 1:
        return x[None]
    if x.ndim != 2:
        raise ValueError(f"expected (T, D) or (D,), got {x.shape}")
    return x


def _sample_cam_present(sample: dict[str, Any], names: tuple[str, ...], cam: str) -> bool:
    if cam not in names:
        return False
    mask = sample.get("camera_mask")
    if mask is None:
        return True
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask).reshape(-1)
    index = names.index(cam)
    if index >= arr.size:
        return True
    return bool(arr[index])


def _camera_names(sample: dict[str, Any]) -> tuple[str, ...]:
    keys = sample.get("camera_keys")
    if keys:
        return tuple(keys)
    image = sample["image"]
    n = len(image) if not torch.is_tensor(image) else int(image.shape[0] if image.ndim == 4 else image.shape[1])
    if torch.is_tensor(image) and image.ndim == 6:
        n = int(image.shape[1])
    return tuple(f"cam{i}" for i in range(n))


def pad_loader_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of packed samples, padding T / D / cameras."""
    if not samples:
        raise ValueError("empty sample list")

    union_cams: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for cam in _camera_names(sample):
            if cam not in seen:
                seen.add(cam)
                union_cams.append(cam)

    per_cam_frames: list[dict[str, np.ndarray]] = []
    t_hist = 1
    height = width = 1
    ch = 3
    for sample in samples:
        names = _camera_names(sample)
        images = sample["image"]
        if torch.is_tensor(images):
            images = images.numpy()
        frames: dict[str, np.ndarray] = {}
        for i, cam in enumerate(names):
            thwc = _as_thwc(np.asarray(images[i]))
            frames[cam] = thwc
            t_hist = max(t_hist, int(thwc.shape[0]))
            height, width, ch = int(thwc.shape[1]), int(thwc.shape[2]), int(thwc.shape[3])
        per_cam_frames.append(frames)

    bsz = len(samples)
    n_cams = len(union_cams)
    image = np.zeros((bsz, n_cams, t_hist, height, width, ch), dtype=np.uint8)
    camera_mask = np.zeros((bsz, n_cams), dtype=bool)
    for b, frames in enumerate(per_cam_frames):
        for c, cam in enumerate(union_cams):
            if cam not in frames:
                continue
            thwc = frames[cam]
            t = thwc.shape[0]
            image[b, c, t_hist - t : t_hist] = thwc
            camera_mask[b, c] = _sample_cam_present(samples[b], _camera_names(samples[b]), cam)

    actions = [_as_2d(np.asarray(s["action"])) for s in samples]
    t_act = max(a.shape[0] for a in actions)
    d_act = max(a.shape[1] for a in actions)
    action = np.zeros((bsz, t_act, d_act), dtype=np.float32)
    action_mask = np.zeros((bsz, t_act, d_act), dtype=bool)
    for b, a in enumerate(actions):
        t, d = a.shape
        action[b, :t, :d] = a
        action_mask[b, :t, :d] = True

    tags = [str(s.get("robot_tag", "")) for s in samples]
    embodiment_id = np.array(
        [
            int(s["embodiment_id"]) if "embodiment_id" in s else embodiment_id_from_tag(tags[i])
            for i, s in enumerate(samples)
        ],
        dtype=np.int64,
    )

    out: dict[str, Any] = {
        "image": torch.from_numpy(np.ascontiguousarray(image)),
        "action": torch.from_numpy(np.ascontiguousarray(action)),
        "action_mask": torch.from_numpy(action_mask),
        "camera_mask": torch.from_numpy(camera_mask),
        "camera_keys": tuple(union_cams),
        "lang": [s["lang"] for s in samples],
        "robot_tag": tags,
        "embodiment_id": torch.from_numpy(embodiment_id),
    }
    if "state" in samples[0]:
        states = [_as_2d(np.asarray(s["state"])) for s in samples]
        t_st = max(x.shape[0] for x in states)
        d_st = max(x.shape[1] for x in states)
        state = np.zeros((bsz, t_st, d_st), dtype=np.float32)
        state_mask = np.zeros((bsz, t_st, d_st), dtype=bool)
        for b, x in enumerate(states):
            t, d = x.shape
            state[b, :t, :d] = x
            state_mask[b, :t, :d] = True
        out["state"] = torch.from_numpy(np.ascontiguousarray(state))
        out["state_mask"] = torch.from_numpy(state_mask)
    return out


def attach_batch_ids_and_masks(batch: dict[str, Any]) -> dict[str, Any]:
    """Fill embodiment_id / masks on an already-stacked (homogeneous) batch in place."""
    out = batch
    tags = list(out.get("robot_tag") or [])
    bsz = int(out["action"].shape[0]) if torch.is_tensor(out.get("action")) else len(tags)
    if "embodiment_id" not in out:
        ids = [embodiment_id_from_tag(tags[i] if i < len(tags) else None) for i in range(bsz)]
        out["embodiment_id"] = torch.as_tensor(ids, dtype=torch.int64)
    action = out["action"]
    if not torch.is_tensor(action):
        action = torch.as_tensor(action)
        out["action"] = action
    if "action_mask" not in out:
        out["action_mask"] = torch.ones(action.shape, dtype=torch.bool)
    image = out.get("image")
    if image is not None and "camera_mask" not in out:
        n_cams = int(image.shape[1]) if torch.is_tensor(image) else len(image[0])
        out["camera_mask"] = torch.ones(bsz, n_cams, dtype=torch.bool)
    return out


def collate_fn(batch: list[dict] | dict) -> dict:
    """Collate samples into a training batch.

    ``__getitems__`` may already return a batched dict; attach masks / ids.
    Otherwise pad a list of per-sample dicts (heterogeneous mixture).
    """
    if isinstance(batch, dict):
        return attach_batch_ids_and_masks(batch)
    return pad_loader_batch(batch)


def _iter_mmap_stores(dataset: Any):
    stores = getattr(dataset, "_mmap_stores", None)
    if isinstance(stores, dict):
        yield from stores.values()
    elif stores:
        yield from stores
    for sub in getattr(dataset, "datasets", []) or []:
        yield from _iter_mmap_stores(sub)


def dataloader_worker_init_fn(_worker_id: int) -> None:
    """After DataLoader fork: pin threads and recreate the JPEG decode pool."""
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    for store in _iter_mmap_stores(info.dataset):
        tune = getattr(store, "tune_decode_workers_for_loaders", None)
        if tune is not None:
            tune(max(1, info.num_workers))
        pool = getattr(store, "_get_decode_pool", None)
        if pool is None:
            continue
        decode_pool = pool()
        if decode_pool is not None:
            decode_pool.submit(lambda: None).result(timeout=15)
