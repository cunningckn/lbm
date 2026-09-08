from lbm.utils.bench import CallStats, time_calls


def test_call_stats_hz_and_samples():
    stats = CallStats(elapsed_s=0.5, iters=10, batch_size=2)
    assert stats.hz == 20.0
    assert stats.samples_per_sec == 40.0
    assert stats.ms_per_call == 50.0


def test_time_calls_counts_iters():
    n = {"i": 0}

    def fn():
        n["i"] += 1

    stats = time_calls(fn, warmup=2, iters=5, batch_size=3, device="cpu")
    assert n["i"] == 7
    assert stats.iters == 5
    assert stats.batch_size == 3
    assert stats.hz > 0
    assert stats.samples_per_sec == stats.hz * 3
