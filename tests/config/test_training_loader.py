import pytest
import torch.distributed as dist

from lbm.config import TrainConfig
from lbm.training_loader import make_loader


@pytest.mark.parametrize("size", [0, 3, 4])
def test_training_loader_rejects_insufficient_single_rank_data(size):
    config = TrainConfig(batch_size=4, num_workers=0)
    if size < 4:
        with pytest.raises(ValueError, match="Training loader has no batches.*batch_size=4"):
            make_loader(range(size), config=config, distributed=False, train=True, collate_fn=None)
    else:
        loader, _ = make_loader(range(size), config=config, distributed=False, train=True, collate_fn=None)
        assert len(list(loader)) == 1


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("size", [0, 1, 7, 8])
def test_training_loader_checks_distributed_samples_per_rank(monkeypatch, rank, size):
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    config = TrainConfig(batch_size=4, num_workers=0)
    if size < 8:
        with pytest.raises(ValueError, match="Training loader has no batches.*world_size=2"):
            make_loader(range(size), config=config, distributed=True, train=True, collate_fn=None)
    else:
        loader, _ = make_loader(range(size), config=config, distributed=True, train=True, collate_fn=None)
        assert len(list(loader)) == 1


def test_validation_loader_can_be_empty():
    loader, _ = make_loader(
        range(0), config=TrainConfig(batch_size=4, num_workers=0),
        distributed=False, train=False, collate_fn=None,
    )
    assert len(loader) == 0


def test_training_loader_allows_explicit_partial_batches():
    loader, _ = make_loader(
        range(1), config=TrainConfig(batch_size=4, num_workers=0),
        distributed=False, train=True, collate_fn=None, drop_last=False,
    )
    assert len(list(loader)) == 1


@pytest.mark.parametrize("workers", [-1, 1.5, True, "2"])
def test_loader_rejects_invalid_worker_count(workers):
    config = TrainConfig(fake_data=True, num_workers=workers)
    with pytest.raises(ValueError, match="num_workers must be a non-negative integer"):
        make_loader(range(4), config=config, distributed=False, train=True, collate_fn=None)


@pytest.mark.parametrize("factor", [0, None])
def test_single_process_loader_ignores_worker_only_settings(factor):
    config = TrainConfig(fake_data=True, num_workers=0)
    config.data.prefetch_factor = factor
    config.data.persistent_workers = True
    config.data.pin_memory = False
    loader, _ = make_loader(range(4), config=config, distributed=False, train=True, collate_fn=None)
    assert loader.prefetch_factor is None
    assert not loader.persistent_workers
    assert sorted(next(iter(loader)).tolist()) == list(range(4))


@pytest.mark.parametrize("factor", [1, 2, 32, None])
def test_multiworker_loader_accepts_supported_prefetch(factor):
    config = TrainConfig(fake_data=True, num_workers=2)
    config.data.prefetch_factor = factor
    config.data.persistent_workers = False
    loader, _ = make_loader(range(4), config=config, distributed=False, train=True, collate_fn=None)
    assert loader.prefetch_factor == (2 if factor is None else factor)
