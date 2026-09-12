"""Episode isolation and balanced, complete validation sampling."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from lbm.training_loader import make_loader


def episode_ids(source):
    """Identify records by canonical location and record metadata (e.g. parquet row range)."""
    if source.records:
        return [hashlib.sha256(json.dumps(
            [str(Path(r.path).resolve()), r.kind, r.extra], sort_keys=True, default=str
        ).encode()).hexdigest() for r in source.records]
    return [f'memory:{id(e)}' for e in source.episodes]


def episode_view(source, indices):
    view = copy.copy(source)
    view._vec_cache, view._mmap_stores, view._parquet_stores = {}, {}, {}
    view._index = None
    if source.records:
        view.records = [source.records[i] for i in indices]
        lengths = [r.n_frames for r in view.records]
    else:
        view.episodes = [source.episodes[i] for i in indices]
        lengths = [len(e.action) for e in view.episodes]
    view._offsets = np.cumsum(lengths, dtype=np.int64)
    return view


def prepare_validation(train, validation, config):
    from lbm.dataloader.custom import CustomMixtureDataset
    from lbm.utils.preprocess import compute_norm_stats

    if config.data.val_fraction:
        pairs, heldout = [], []
        for source, weight in zip(train.datasets, train.weights, strict=True):
            count = len(source.records or source.episodes)
            if count < 2:
                raise ValueError(f'{source.spec.name}: episode holdout requires at least two episodes')
            order = np.random.default_rng(config.seed).permutation(count)
            cut = min(count - 1, max(1, int(count * config.data.val_fraction)))
            tr, va = episode_view(source, order[cut:]), episode_view(source, order[:cut])
            # Never reuse full-corpus statistics after splitting.
            tr.norm_stats = va.norm_stats = compute_norm_stats(tr, progress=False)['norm_stats']
            pairs.append((tr, weight))
            heldout.append((va, 1.0))
        train = CustomMixtureDataset(pairs, seed=config.seed)
        validation = CustomMixtureDataset(heldout, mode='val', seed=config.seed)
    if validation:
        ids = {identity for ds in train.datasets for identity in episode_ids(ds)}
        if any(ids.intersection(episode_ids(ds)) for ds in validation.datasets):
            raise ValueError('training and validation episodes overlap')
        sources = {ds.spec.name: ds for ds in train.datasets}
        for ds in validation.datasets:
            if ds.spec.name not in sources:
                raise ValueError(f'validation source {ds.spec.name} has no training normalization contract')
            ds.norm_stats = sources[ds.spec.name].norm_stats
    return train, validation


def balanced_indices(source, take):
    offsets = getattr(source, '_episode_offsets', None)
    if offsets is None and (getattr(source, 'records', None) or getattr(source, 'episodes', None)):
        offsets = source._offsets
    if offsets is None:
        return np.linspace(0, len(source) - 1, take, dtype=np.int64).tolist()
    starts = np.concatenate(([0], offsets[:-1]))
    lengths = np.asarray(offsets) - starts
    active = np.flatnonzero(lengths > 0)
    if len(active) > take:
        active = active[np.linspace(0, len(active) - 1, take, dtype=int)]
    counts = np.zeros(len(lengths), dtype=np.int64)
    left = take
    while left:
        for episode in active:
            if counts[episode] < lengths[episode]:
                counts[episode] += 1
                left -= 1
                if not left:
                    break
    return [int(starts[e] + i) for e in active
            for i in np.linspace(0, lengths[e] - 1, counts[e], dtype=np.int64)]


def validation_loaders(dataset, config, collate_fn, *, rank=0, world=1):
    """Balance sources, spread samples across episodes, pad ranks without counting duplicates."""
    if not dataset or config.val_batches == 0:
        return {}
    sources = getattr(dataset, 'datasets', None) or [dataset]
    budget = config.val_batches * config.batch_size * world
    if budget < len(sources):
        raise ValueError('validation budget must cover every source; increase val_batches')
    loaders = {}
    for i, source in enumerate(sources):
        take = min(len(source), budget // len(sources) + (i < budget % len(sources)))
        selected = balanced_indices(source, take)
        local = selected[rank::world]
        valid = len(local)
        # FSDP needs equal forward counts on all ranks, including a small tail.
        local.extend([selected[0]] * ((take + world - 1) // world - valid))
        subset = Subset(source, local)
        subset.valid_count = valid
        name = getattr(getattr(source, 'spec', None), 'name', 'all')
        loaders[f'{i}:{name}'], _ = make_loader(
            subset, config=config, distributed=False, train=False, collate_fn=collate_fn,
            drop_last=False, num_workers=min(2, config.num_workers),
        )
    return loaders


def dataset_fingerprint(dataset):
    """Detect reordered/replaced records and changed normalization even at equal length."""
    from lbm.utils.preprocess import _jsonable

    sources = []
    for source in dataset.datasets:
        files = []
        for record in source.records:
            path = Path(record.path)
            stat = path.stat()
            files.append((stat.st_size, stat.st_mtime_ns))
        sources.append(dict(episodes=episode_ids(source), files=files,
                            normalization=_jsonable(source.norm_stats)))
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
