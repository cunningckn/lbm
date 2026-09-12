from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import default_collate

from lbm.config import TrainConfig
from lbm.training_split import prepare_validation, validation_loaders
from lbm.training_validation import evaluate_actions


class Samples:
    def __init__(self, name, count):
        self.spec = SimpleNamespace(name=name)
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {'actions': torch.ones(1, 1) * (index + 1)}


def test_validation_covers_sources_tail_and_small_rank_without_double_counting():
    cfg = TrainConfig(batch_size=4, num_workers=0, val_batches=20)
    dataset = SimpleNamespace(datasets=[Samples('A', 9), Samples('B', 1)])
    model = torch.nn.Linear(1, 1)
    module = SimpleNamespace(sample_actions=lambda batch, **kw: torch.zeros_like(batch['actions']))
    total = torch.zeros(2, dtype=torch.float64)
    for rank in range(3):
        loaders = validation_loaders(dataset, cfg, default_collate, rank=rank, world=3)
        assert len(loaders) == 2
        total += evaluate_actions(model, module, loaders, lambda raw, **kw: raw,
                                  device='cpu', num_steps=1)
    torch.testing.assert_close(total, torch.tensor([286., 10.], dtype=torch.float64))


def test_validation_disabled_is_empty():
    cfg = TrainConfig(val_batches=0)
    assert validation_loaders(Samples('A', 10), cfg, default_collate) == {}


def test_explicit_episode_overlap_rejected():
    record = SimpleNamespace(path='/demo/episode', kind='demo', extra={})
    source = SimpleNamespace(records=[record], spec=SimpleNamespace(name='A'))
    mixture = SimpleNamespace(datasets=[source])
    with pytest.raises(ValueError, match='overlap'):
        prepare_validation(mixture, mixture, TrainConfig())


def test_episode_holdout_recomputes_train_only_stats():
    from lbm.dataloader.custom import CUSTOM_SPECS, CustomMixtureDataset, CustomSingleDataset
    from lbm.training_split import episode_ids
    from tests.dataloader.test_custom import _episode

    spec = CUSTOM_SPECS['libero']
    source = CustomSingleDataset(spec, [_episode(spec, seed=i) for i in range(5)],
                                 action_mode='abs', norm_stats={'stale': True})
    cfg = TrainConfig()
    cfg.data.val_fraction = .4
    train, validation = prepare_validation(CustomMixtureDataset([(source, 1)]), (), cfg)
    tr, va = train.datasets[0], validation.datasets[0]
    assert len(tr.episodes) == 3 and len(va.episodes) == 2
    assert not set(episode_ids(tr)) & set(episode_ids(va))
    assert 'stale' not in tr.norm_stats
    assert va.norm_stats is tr.norm_stats
    assert tr._locate(len(tr) - 1) == (2, 11)


def test_validation_covers_short_episodes_between_long_ones():
    import numpy as np

    from lbm.training_split import balanced_indices

    source = Samples('A', 10003)
    source._episode_offsets = np.array([1, 2, 10002, 10003])
    assert balanced_indices(source, 4) == [0, 1, 2, 10002]
    indices = balanced_indices(source, 20)
    assert len(indices) == len(set(indices)) == 20
    assert {0, 1, 10002} <= set(indices)
