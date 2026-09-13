"""Deterministic sample RNG and cursor-addressable batches, independent of worker prefetch."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class SeededDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitems__(self, keys):
        # Cached features have no random transforms. Other datasets must retain
        # per-sample RNG isolation, even when they implement their own bulk API.
        from lbm.training_features import FeatureDataset

        if isinstance(self.dataset, FeatureDataset):
            return self.dataset.__getitems__([key[2] for key in keys])
        return [self[key] for key in keys]

    def __getitem__(self, key):
        epoch, position, index = key
        seed = int(np.random.SeedSequence([self.seed, epoch, position]).generate_state(1)[0])
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            # Avoid touching CUDA RNG from worker processes.
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            return self.dataset[index]
        finally:
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)


class CursorBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, *, seed, rank=0, world=1, drop_last=True):
        self.dataset, self.batch_size = dataset, batch_size
        self.seed, self.rank, self.world, self.drop_last = seed, rank, world, drop_last
        self.epoch, self.start_batch = 0, 0
        self.num_replicas = world

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        size = len(self.dataset)
        width = self.batch_size * self.world
        return size // width if self.drop_last else (size + width - 1) // width

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        weights = getattr(self.dataset, 'weights', [])
        weighted = bool(weights) and len(set(weights)) > 1
        if weighted:
            probs = torch.tensor([w * len(ds) for w, ds in zip(
                weights, self.dataset.datasets, strict=True)], dtype=torch.float64)
            offsets = np.concatenate(([0], np.cumsum([len(ds) for ds in self.dataset.datasets])))
            sources = torch.multinomial(probs, len(self.dataset), replacement=True, generator=generator)
            draws = torch.rand(len(self.dataset), generator=generator).numpy()
            source_ids = sources.numpy()
            order = offsets[source_ids] + (draws * np.diff(offsets)[source_ids]).astype(np.int64)
        else:
            order = torch.randperm(len(self.dataset), generator=generator).numpy()
        width = self.batch_size * self.world
        for batch in range(self.start_batch, len(self)):
            first = batch * width + self.rank * self.batch_size
            positions = range(first, min(first + self.batch_size, len(order)))
            indices = [(self.epoch, pos, int(order[pos])) for pos in positions]
            if indices:
                yield indices
