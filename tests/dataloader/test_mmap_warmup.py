"""Sampler-ahead mmap warmup follows the index cursor, not a static shard."""

from __future__ import annotations

from concurrent.futures import Future

from lbm.dataloader.mmap_warmup import (
    indices_after,
    mmap_warmup_window,
    schedule_mmap_ahead,
    warmup_first_window,
)


class _DummyStore:
    enabled = True

    def __init__(self) -> None:
        self.submitted: list = []

    def _get_prefetch_pool(self):
        store = self

        class _Pool:
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)
                fut = Future()
                fut.set_result(None)
                store.submitted.append(True)
                return fut

        return _Pool()


class _DummyDataset:
    def __init__(self, n: int, *, window: int = 8, order: list[int] | None = None) -> None:
        self._n = n
        self._mmap_warmup_samples = window
        self.data_cfg = {"mmap_warmup_samples": window}
        self._mmap_video_store = _DummyStore()
        self._epoch_indices = order
        self.prefaulted: list[int] = []

    def __len__(self) -> int:
        return self._n


def test_indices_after_sequential() -> None:
    ds = _DummyDataset(20, window=8)
    assert indices_after(ds, [0, 1, 2, 3], 8) == list(range(4, 12))
    assert indices_after(ds, [16, 17, 18, 19], 8) == []


def test_indices_after_shuffled_epoch_order() -> None:
    order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ds = _DummyDataset(10, window=3, order=order)
    # Batch is consecutive positions in the permutation, not consecutive ids.
    assert indices_after(ds, [9, 8], 3) == [7, 6, 5]


def test_warmup_window_prefers_data_cfg() -> None:
    ds = _DummyDataset(4, window=2)
    ds.data_cfg = {"mmap_warmup_samples": 12}
    ds._mmap_warmup_samples = 99
    assert mmap_warmup_window(ds) == 12


def test_schedule_mmap_ahead_slides_and_skips_already_covered(monkeypatch) -> None:
    ds = _DummyDataset(32, window=8)
    calls: list[list[int]] = []

    def _capture_grouped(dataset, index: int) -> None:
        calls[-1].append(index)

    monkeypatch.setattr("lbm.dataloader.mmap_warmup.prefault_one_index", _capture_grouped)

    calls.append([])
    schedule_mmap_ahead(ds, [0, 1, 2, 3])
    assert calls[-1] == list(range(4, 12))
    assert ds._mmap_ahead_hi == 11

    calls.append([])
    schedule_mmap_ahead(ds, [4, 5, 6, 7])
    # Window is still 8 ahead of this batch (8..15); 4..11 were already marked.
    assert calls[-1] == list(range(12, 16))
    assert ds._mmap_ahead_hi == 15

    calls.append([])
    schedule_mmap_ahead(ds, [4, 5, 6, 7])
    assert calls[-1] == []


def test_first_window_is_sampler_prefix_not_strided_dataset_ids(monkeypatch) -> None:
    order = [10, 20, 30, 40, 50, 60, 70, 80]
    ds = _DummyDataset(8, window=4, order=order)
    seen: list[int] = []

    def _capture(dataset, index: int) -> None:
        seen.append(index)

    monkeypatch.setattr("lbm.dataloader.mmap_warmup.prefault_one_index", _capture)
    warmup_first_window(ds, worker_id=0, num_workers=2)
    warmup_first_window(ds, worker_id=1, num_workers=2)

    # Shared-queue prefix is order[:window], split across workers — not 0,2,4,... of the dataset.
    assert sorted(seen) == [10, 20, 30, 40]
    assert ds._mmap_ahead_hi == 3
