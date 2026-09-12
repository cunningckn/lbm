import random

import numpy as np
import torch

from lbm.config import TrainConfig
from lbm.dataloader.custom.dataset import CustomMixtureDataset
from lbm.training_loader import make_loader
from lbm.training_sampler import CursorBatchSampler


class Jitter:
    def __len__(self):
        return 32

    def __getitem__(self, index):
        return index, random.random(), np.random.rand(), torch.rand(()).item()


def test_workers_and_cursor_preserve_samples_without_replaying_reads():
    def collect(workers, cursor):
        cfg = TrainConfig(batch_size=4, num_workers=workers, fake_data=True)
        loader, sampler = make_loader(Jitter(), config=cfg, distributed=False, train=True, collate_fn=None)
        sampler.set_epoch(3)
        sampler.start_batch = cursor
        return [torch.stack(batch, dim=1) for batch in loader]
    full = collect(0, 0)
    resumed = collect(2, 5)
    torch.testing.assert_close(torch.cat(full[5:]), torch.cat(resumed), rtol=0, atol=0)


def test_weighted_sampling_and_compact_mixture_index():
    mix = CustomMixtureDataset([(range(1000), 100), (range(1000), 1)])
    assert mix._offsets.nbytes == 16
    assert not hasattr(mix, '_map')
    sampler = CursorBatchSampler(mix, 10, seed=123)
    draws = [key[2] for batch in sampler for key in batch]
    ratio = sum(i < 1000 for i in draws) / len(draws)
    assert .975 < ratio < .999
    assert mix[999] == 999 and mix[1000] == 0 and mix[-1] == 999


def test_distributed_sampling_partitions_without_overlap():
    a, b = [CursorBatchSampler(range(32), 4, seed=7, rank=r, world=2) for r in range(2)]
    left = {key[2] for batch in a for key in batch}
    right = {key[2] for batch in b for key in batch}
    assert not left & right
    assert left | right == set(range(32))


def test_cursor_does_not_read_preceding_batches():
    seen = []
    class Recording:
        def __len__(self):
            return 100
        def __getitem__(self, index):
            seen.append(index)
            return index
    cfg = TrainConfig(batch_size=4, num_workers=0, fake_data=True)
    loader, sampler = make_loader(Recording(), config=cfg, distributed=False, train=True, collate_fn=None)
    sampler.start_batch = 24
    assert len(list(loader)) == 1
    assert len(seen) == 4
