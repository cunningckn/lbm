from lbm.utils.mem import (
    process_rss_bytes,
    reclaim_if_over,
    set_worker_rss_limit,
    worker_rss_limit_bytes,
)


def test_process_rss_bytes_positive() -> None:
    assert process_rss_bytes() > 0


def test_worker_rss_limit_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LBM_WORKER_RSS_MB", "512")
    assert worker_rss_limit_bytes(8) == 512 * 1024 * 1024


def test_worker_rss_limit_splits_cgroup(monkeypatch) -> None:
    monkeypatch.delenv("LBM_WORKER_RSS_MB", raising=False)
    monkeypatch.setattr("lbm.utils.mem.cgroup_memory_max_bytes", lambda: 10 * 1024**3)
    assert worker_rss_limit_bytes(4) == int(10 * 1024**3 * 0.8 / 4)


def test_reclaim_if_over_skips_under_limit(monkeypatch) -> None:
    hits = {"n": 0}
    monkeypatch.setattr("lbm.utils.mem.process_rss_bytes", lambda: 100)
    monkeypatch.setattr("lbm.utils.mem.malloc_trim", lambda: hits.__setitem__("n", hits["n"] + 1) or True)
    assert reclaim_if_over(1000) is False
    assert hits["n"] == 0


def test_reclaim_if_over_runs_when_exceeded(monkeypatch) -> None:
    hits = {"n": 0}
    monkeypatch.setattr("lbm.utils.mem.process_rss_bytes", lambda: 2000)
    monkeypatch.setattr("lbm.utils.mem.malloc_trim", lambda: hits.__setitem__("n", hits["n"] + 1) or True)
    monkeypatch.setattr("lbm.utils.mem.gc.collect", lambda: 0)
    assert reclaim_if_over(1000) is True
    assert hits["n"] == 1


def test_set_worker_rss_limit_used_as_default(monkeypatch) -> None:
    set_worker_rss_limit(50)
    try:
        monkeypatch.setattr("lbm.utils.mem.process_rss_bytes", lambda: 80)
        monkeypatch.setattr("lbm.utils.mem.malloc_trim", lambda: True)
        monkeypatch.setattr("lbm.utils.mem.gc.collect", lambda: 0)
        assert reclaim_if_over() is True
    finally:
        set_worker_rss_limit(None)
