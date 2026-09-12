import pytest
import torch

from lbm.training_validation import evaluate_actions


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("failure", [None, "loader", "sample", "reduce"])
def test_validation_restores_mode_even_on_failure(monkeypatch, training, failure):
    class Policy(torch.nn.Module):
        def sample_actions(self, batch, **kwargs):
            assert not self.training
            assert not torch.is_grad_enabled()
            if failure == "sample":
                raise RuntimeError("sample")
            return torch.ones_like(batch["actions"])

    def loader():
        if failure == "loader":
            raise RuntimeError("loader")
        yield {"actions": torch.zeros(1, 2, 3)}

    def reduce(stats, **kwargs):
        raise RuntimeError("reduce")

    monkeypatch.setattr("lbm.training_validation.dist.all_reduce", reduce)
    model = Policy().train(training)
    args = dict(device="cpu", num_steps=2, distributed=failure == "reduce")
    def convert(batch, **kwargs):
        return batch
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            evaluate_actions(model, model, loader(), convert, **args)
    else:
        stats = evaluate_actions(model, model, loader(), convert, **args)
        torch.testing.assert_close(stats, torch.tensor([6., 6.], dtype=torch.float64))
    assert model.training == training


def test_empty_validation_returns_zero_stats():
    model = torch.nn.Linear(1, 1)
    stats = evaluate_actions(model, model, None, None, device="cpu", num_steps=1)
    assert model.training
    torch.testing.assert_close(stats, torch.zeros(2, dtype=torch.float64))
