import numpy as np
import pytest

from lbm.history import ObservationHistory, frame_history, timestamp_history


@pytest.mark.parametrize("origin", [0.0, 1720000000.0])
@pytest.mark.parametrize("fps, frequency", [(15, 10), (5, 10), (30, 7)])
def test_offline_online_history_agree_at_fractional_native_stride(fps, frequency, origin):
    frames = frame_history(17, lower=0, fps=fps, length=0.4, frequency=frequency)
    online = timestamp_history(origin + np.arange(18) / fps, now=origin + 17 / fps, length=0.4, frequency=frequency)
    np.testing.assert_array_equal(frames.indices, online.indices)
    np.testing.assert_array_equal(frames.valid, online.valid)
    np.testing.assert_allclose(frames.offsets, online.offsets, atol=3e-7 if origin else 1e-12)
    if (fps, frequency) == (15, 10):
        assert frames.indices.tolist() == [12, 14, 15, 17]


def test_missing_history_is_masked_and_never_reads_a_future_observation():
    frames = frame_history(5, lower=4, fps=10, length=0.4, frequency=10)
    assert frames.indices.tolist() == [4, 4, 4, 5]
    assert frames.valid.tolist() == [False, False, True, True]
    sample = timestamp_history([0.0, 0.35, 0.4], now=0.4, length=0.4, frequency=10)
    assert sample.indices.tolist() == [0, 0, 0, 2]
    assert sample.valid.tolist() == [True, False, False, True]
    with pytest.raises(ValueError, match="future"):
        timestamp_history([0.0, 0.5], now=0.4, length=0.4, frequency=10)


def test_history_is_bounded_and_context_changes_reset_old_observations():
    history = ObservationHistory(horizon=1.0, capacity=3)
    for i in range(10):
        history.append(i / 10, i, context="episode-a")
    assert len(history) == 3
    history.append(0.9, 99, context="episode-a")
    values, selected = history.select(length=0.3, frequency=10)
    assert values == [7, 8, 99] and selected.valid.all()
    with pytest.raises(ValueError, match="decreased"):
        history.append(0.8, 0, context="episode-a")
    history.append(0.0, 123, context="episode-b")
    values, selected = history.select(length=0.3, frequency=10)
    assert values == [123] * 3 and selected.valid.tolist() == [False, False, True]
    history.reset()
    with pytest.raises(ValueError, match="empty"):
        history.select(length=0.3, frequency=10)
