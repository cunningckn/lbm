"""Loader construction shared by training and validation.

Keeps sampling, worker setup and empty-loader checks separate from the runner.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler

from lbm.config import TrainConfig, validate_loader_config


def make_loader(
    dataset,
    *,
    config: TrainConfig,
    distributed: bool,
    train: bool,
    collate_fn,
    drop_last: bool = True,
    num_workers: int | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Build a train or validation loader and return its distributed sampler.

    Training uses a dedicated seeded generator so the runner can replay epoch
    order on resume. Callers own sampler.set_epoch() and loader iteration.
    """
    sampler = None
    workers = config.num_workers if num_workers is None else num_workers
    errors = validate_loader_config(config.data, num_workers=workers)
    if errors:
        raise ValueError("Invalid loader config: " + "; ".join(errors))
    if distributed:
        sampler = DistributedSampler(
            dataset, shuffle=train, seed=config.seed, drop_last=drop_last
        )
    samples_per_rank = len(sampler) if sampler is not None else len(dataset)
    if train and (samples_per_rank == 0 or (drop_last and samples_per_rank < config.batch_size)):
        world_size = sampler.num_replicas if sampler is not None else 1
        raise ValueError(
            "Training loader has no batches: "
            f"dataset_size={len(dataset)}, samples_per_rank={samples_per_rank}, "
            f"batch_size={config.batch_size}, world_size={world_size}, drop_last={drop_last}. "
            "Reduce batch_size or world_size, or provide more training samples."
        )
    kwargs: dict = {
        "batch_size": config.batch_size,
        "sampler": sampler,
        "shuffle": sampler is None and train,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": config.data.pin_memory,
        "drop_last": drop_last,
    }
    if train:
        kwargs["generator"] = torch.Generator().manual_seed(config.seed)
    if workers > 0:
        kwargs["persistent_workers"] = config.data.persistent_workers
        kwargs["prefetch_factor"] = config.data.prefetch_factor
        if not config.fake_data:
            from lbm.dataloader.pad import dataloader_worker_init_fn

            kwargs["worker_init_fn"] = dataloader_worker_init_fn
    if train:
        from lbm.training_sampler import CursorBatchSampler, SeededDataset

        rank = sampler.rank if sampler else 0
        world = sampler.num_replicas if sampler else 1
        sampler = CursorBatchSampler(dataset, config.batch_size, seed=config.seed,
                                     rank=rank, world=world, drop_last=drop_last)
        for key in ('batch_size', 'sampler', 'shuffle', 'drop_last'):
            kwargs.pop(key)
        kwargs['batch_sampler'] = sampler
        dataset = SeededDataset(dataset, config.seed)
    return DataLoader(dataset, **kwargs), sampler
