import numpy as np
import pytest

from lbm.utils.preprocess import _Moments
from lbm.utils.quantiles import DiskQuantiles


@pytest.mark.parametrize("n", [1, 2, 100, 4097])
def test_disk_quantiles_match_numpy_across_blocks(tmp_path, n):
    rng = np.random.default_rng(23)
    values = rng.normal(size=(n, 5)).astype(np.float32)
    values[:, 1] = 0
    values[0, 2] = 1e30
    values[-1, 3] = -1e30
    q = DiskQuantiles(directory=tmp_path, block_rows=31)
    try:
        for start in range(0, n, 17):
            q.append(values[start:start + 17])
        ps = [0, .01, .25, .5, .99, 1]
        np.testing.assert_allclose(q.quantiles(ps), np.quantile(values, ps, axis=0), rtol=1e-7, atol=1e-7)
        q.append(values[:1])
        np.testing.assert_allclose(q.quantiles(ps), np.quantile(np.concatenate((values, values[:1])), ps, axis=0),
                                   rtol=1e-7, atol=1e-7)
    finally:
        q.close()
    assert q.file.closed
    assert not list(tmp_path.iterdir())


def test_moments_cleanup_when_quantiles_fail(tmp_path, monkeypatch):
    with pytest.raises(OSError):
        with _Moments(scratch_dir=tmp_path, block_rows=8) as m:
            m.update(np.ones((21, 3)))
            def fail(*args):
                raise OSError("scratch disk failed")
            monkeypatch.setattr(m.quantiles, "quantiles", fail)
            m.as_dict()
    assert m.quantiles.file.closed
    assert not list(tmp_path.iterdir())


def test_moments_statistics_match_direct_computation(tmp_path):
    values = np.random.default_rng(11).normal(size=(1003, 7)).astype(np.float32)
    with _Moments(scratch_dir=tmp_path, block_rows=19) as m:
        m.update(values)
        stats = m.as_dict()
    for key, expected in (("mean", values.mean(0)), ("std", values.std(0)),
                          ("q01", np.quantile(values, .01, axis=0)),
                          ("q99", np.quantile(values, .99, axis=0))):
        np.testing.assert_allclose(stats[key], expected, atol=1e-6)
    assert stats["count"] == len(values)
