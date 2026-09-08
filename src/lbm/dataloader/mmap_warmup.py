"""Sampler-ahead JPEG page-in. Does not wait on the current batch.

``mmap_warmup_samples`` is a sliding window in sampler order: after each
``__getitems__`` the next ``window`` indices are scheduled on a side thread.
Only newly uncovered indices are touched, so later data stays warm as the
cursor moves. Default DataLoader uses a shared index queue, so this follows
the global sampler, not ``prefix[worker_id::N]``.
"""

from __future__ import annotations


def mmap_warmup_window(dataset) -> int:
    cfg = getattr(dataset, "data_cfg", None) or {}
    try:
        k = int(cfg.get("mmap_warmup_samples", 0) or 0)
    except (TypeError, ValueError):
        k = 0
    if k <= 0:
        k = int(getattr(dataset, "_mmap_warmup_samples", 0) or 0)
    return max(0, k)


def epoch_pos_map(dataset) -> dict[int, int] | None:
    order = getattr(dataset, "_epoch_indices", None)
    if order is None:
        return None
    pos = getattr(dataset, "_epoch_pos", None)
    if pos is None:
        pos = {int(idx): i for i, idx in enumerate(order)}
        dataset._epoch_pos = pos
    return pos


def indices_after(dataset, current: list[int], window: int) -> list[int]:
    """Next ``window`` sampler indices after this batch."""
    n = len(dataset)
    if window <= 0 or n <= 0 or not current:
        return []
    order = getattr(dataset, "_epoch_indices", None)
    if order is None:
        start = max(int(i) for i in current) + 1
        return list(range(start, min(start + window, n)))
    pos = epoch_pos_map(dataset)
    if pos is None:
        return []
    positions = [pos[int(i)] for i in current if int(i) in pos]
    if not positions:
        return []
    cur_max = max(positions)
    return [int(order[p]) for p in range(cur_max + 1, min(cur_max + 1 + window, n))]


def peek_trajectory_data(dataset, trajectory_id: int):
    """Load episode parquet without writing ``curr_traj_*`` (prefetch-thread safe)."""
    load = getattr(dataset, "peek_trajectory_data", None)
    if callable(load):
        return load(int(trajectory_id))
    return dataset.get_trajectory_data(int(trajectory_id))


def prefault_one_index(dataset, index: int) -> None:
    """Page-in JPEG ranges for one sampler index. Same windows ``__getitems__`` will decode."""
    if hasattr(dataset, "_resolve_sample"):
        sub, trajectory_id, base = dataset._resolve_sample(int(index))
        traj_data = peek_trajectory_data(sub, int(trajectory_id))
        packed = sub._mmap_video_items(int(trajectory_id), int(base), traj_data=traj_data)
        if packed is not None:
            sub._mmap_video_store.prefault_items(packed[0])
        return
    store = getattr(dataset, "_mmap_video_store", None)
    steps = getattr(dataset, "all_steps", None)
    if store is None or steps is None or not store.enabled:
        return
    trajectory_id, base = steps[int(index)]
    traj_data = peek_trajectory_data(dataset, int(trajectory_id))
    packed = dataset._mmap_video_items(int(trajectory_id), int(base), traj_data=traj_data)
    if packed is not None:
        store.prefault_items(packed[0])


def _prefetch_store(dataset):
    store = getattr(dataset, "_mmap_video_store", None)
    if store is not None and store.enabled:
        return store
    for sub in getattr(dataset, "datasets", []) or []:
        store = getattr(sub, "_mmap_video_store", None)
        if store is not None and store.enabled:
            return store
    return None


def schedule_mmap_ahead(dataset, current_indices: list[int]) -> None:
    """Keep a sliding window of upcoming sampler indices warm. Does not wait."""
    window = mmap_warmup_window(dataset)
    store = _prefetch_store(dataset)
    if window <= 0 or store is None:
        return
    nxt = indices_after(dataset, current_indices, window)
    if not nxt:
        return
    hi = int(getattr(dataset, "_mmap_ahead_hi", -1))
    pos = epoch_pos_map(dataset)
    fresh: list[int] = []
    new_hi = hi
    for idx in nxt:
        p = pos[idx] if pos is not None else idx
        if p > hi:
            fresh.append(idx)
            if p > new_hi:
                new_hi = p
    if not fresh:
        return
    dataset._mmap_ahead_hi = new_hi

    def _run(indices: list[int] = fresh) -> None:
        for sample_index in indices:
            prefault_one_index(dataset, sample_index)

    store._get_prefetch_pool().submit(_run)


def warmup_first_window(dataset, worker_id: int, num_workers: int) -> None:
    """Page-in the first in-flight sampler window, sharded across workers.

    The shared index queue pops the sampler prefix first, so this is the set
    that will actually be fetched. Later batches are covered by ``schedule_mmap_ahead``.
    """
    window = mmap_warmup_window(dataset)
    if window <= 0:
        return
    n = len(dataset)
    window = min(window, n)
    order = getattr(dataset, "_epoch_indices", None)
    first = [int(order[i]) for i in range(window)] if order is not None else list(range(window))
    workers = max(1, num_workers)
    mine = first[worker_id::workers]
    for index in mine:
        prefault_one_index(dataset, index)
    dataset._mmap_ahead_hi = window - 1
