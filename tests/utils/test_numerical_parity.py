import pytest
import torch
from benchmarks.numerical_parity import aggregate, summarize, tensor_stats


def test_update_error_is_not_hidden_by_large_parameter_norm():
    before = torch.tensor([1000., 1000.])
    actual, reference = before + torch.tensor([1., -1.]), before + torch.tensor([1., 1.])
    parameters = summarize(tensor_stats(actual, reference, chunk=1))
    updates = summarize(tensor_stats(actual, reference, actual_before=before, reference_before=before, chunk=1))
    assert parameters['relative_l2'] < .002
    assert updates['relative_l2'] == pytest.approx(2**.5)
    assert updates['cosine'] == 0


def test_aggregate_error_weights_tensor_size_and_handles_zero_reference():
    rows = [tensor_stats(torch.tensor([2., 2.]), torch.ones(2)),
            tensor_stats(torch.zeros(1), torch.zeros(1))]
    assert aggregate(rows)['relative_l2'] == 1.
    assert summarize(rows[1])['relative_l2'] is None
    assert summarize(rows[1])['cosine'] is None
    with pytest.raises(ValueError, match='nonfinite'):
        tensor_stats(torch.tensor([float('nan')]), torch.zeros(1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_gpu_reductions_match_cpu_for_cross_device_update_snapshots():
    generator = torch.Generator().manual_seed(19)
    before = torch.randn(4097, generator=generator)
    reference = before + torch.randn(4097, generator=generator) * .001
    actual = reference + torch.randn(4097, generator=generator) * .0001
    kwargs = dict(actual_before=before, reference_before=before, chunk=127)
    expected = tensor_stats(actual, reference, **kwargs)
    observed = tensor_stats(actual.cuda(), reference, **kwargs)
    assert observed == pytest.approx(expected, rel=1e-12, abs=1e-14)
