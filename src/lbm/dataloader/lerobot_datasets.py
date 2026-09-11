# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from lbm.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from lbm.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    EmbodimentTag,
)

logger = logging.getLogger(__name__)


def _as_thwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        return arr[None]
    if arr.ndim != 4:
        raise ValueError(f"expected image (H,W,3) or (T,H,W,3), got {arr.shape}")
    return arr


def dataloader_worker_init_fn(_worker_id: int) -> None:
    """Pin OpenCV / PyTorch to 1 thread; build JPEG decode pool before the first batch."""
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    stores = []
    store = getattr(dataset, "_mmap_video_store", None)
    if store is not None:
        stores.append(store)
    for sub in getattr(dataset, "datasets", []) or []:
        store = getattr(sub, "_mmap_video_store", None)
        if store is not None:
            stores.append(store)
    for store in stores:
        tune = getattr(store, "tune_decode_workers_for_loaders", None)
        if tune is not None:
            tune(max(1, info.num_workers))
        pool = store._get_decode_pool()
        if pool is not None:
            pool.submit(lambda: None).result(timeout=15)


def collate_fn(batch: list[dict] | dict) -> dict:
    """Collate samples into a training batch.

    The mmap dataset's ``__getitems__`` already returns a batched dict
    (image is ``(B, C, T, H, W, 3)`` uint8); pass that through.

    The fallback path stacks a list of per-sample dicts.
    """
    if isinstance(batch, dict):
        return batch

    first_image = batch[0]["image"]
    n_cams = len(first_image)
    t, h, w, ch = _as_thwc(first_image[0]).shape
    image = torch.empty((len(batch), n_cams, t, h, w, ch), dtype=torch.uint8)
    if torch.utils.data.get_worker_info() is not None:
        image.share_memory_()
    dest = image.numpy()

    actions = []
    langs = []
    tags = []
    states = []
    has_state = "state" in batch[0]
    for i, sample in enumerate(batch):
        for cam_i, cam_img in enumerate(sample["image"]):
            dest[i, cam_i] = _as_thwc(cam_img)
        actions.append(np.asarray(sample["action"]))
        langs.append(sample["lang"])
        tags.append(sample["robot_tag"])
        if has_state:
            states.append(np.asarray(sample["state"]))

    out = {
        "image": image,
        "action": torch.from_numpy(np.ascontiguousarray(np.stack(actions, axis=0))),
        "lang": langs,
        "robot_tag": tags,
    }
    if has_state:
        out["state"] = torch.from_numpy(np.ascontiguousarray(np.stack(states, axis=0)))
    if torch.utils.data.get_worker_info() is not None:
        out["action"].share_memory_()
        if "state" in out:
            out["state"].share_memory_()
    return out


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        print(
            f"Warning: DataConfig for robot_type={robot_type!r} has no embodiment_tag, "
            f"using {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"

    # Opt-in factory hook: a DataConfig may define ``make_dataset(dataset_name=..., **ds_kwargs)``
    # to swap in a custom dataset class (e.g. with per-task filtering / chunk stride).
    # When absent, fall through to the default LeRobotSingleDataset construction below.
    if hasattr(data_config, "make_dataset"):
        return data_config.make_dataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            dataset_name=data_name,
        )

    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(f"[dataloader] Using mixture '{data_mix}': {[(d, w, r) for d, w, r in mixture_spec]}")
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg
                ),
                d_weight,
            )
        )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/simBenchmarks/LIBERO/train_files/bar/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--data_mix", type=str, default=None, help="Override data_mix from config")
    parser.add_argument("--data_root_dir", type=str, default=None, help="Override data_root_dir from config")
    args = parser.parse_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.data_root_dir = Path(vla_dataset_cfg.data_root_dir)
    if args.data_mix is not None:
        vla_dataset_cfg.data_mix = args.data_mix
    if args.data_root_dir is not None:
        vla_dataset_cfg.data_root_dir = Path(args.data_root_dir)

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm

    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 3:
            break
        count += 1
        pass
