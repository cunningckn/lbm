import pytest
import torch

from lbm import train_loop
from lbm.config import TrainConfig
from lbm.train_loop import _action_error_stats


def test_validation_ignores_padded_dimensions_and_timesteps():
    pred = torch.tensor([[[2.0, 100.0], [100.0, 100.0]], [[1.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[[1, 0], [0, 0]], [[1, 1], [0, 0]]], dtype=torch.bool)
    stats = _action_error_stats(pred, torch.zeros_like(pred), mask)
    torch.testing.assert_close(stats, torch.tensor([14.0, 3.0]))


def test_validation_without_mask_matches_mse():
    pred = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    target = torch.ones_like(pred)
    stats = _action_error_stats(pred, target)
    torch.testing.assert_close(stats[0] / stats[1], (pred - target).square().mean())


def test_validation_aggregates_by_valid_elements_across_batches_and_ranks():
    # Rank/batch 0: one valid error of 4; rank/batch 1: three errors of 1.
    pred = torch.tensor([[[2.0, 99.0, 99.0]], [[1.0, 1.0, 1.0]]])
    mask = torch.tensor([[[1, 0, 0]], [[1, 1, 1]]])
    parts = [
        _action_error_stats(p, torch.zeros_like(p), m)
        for p, m in zip(pred.split(1), mask.split(1), strict=True)
    ]
    summed = torch.stack(parts).sum(0)
    torch.testing.assert_close(summed, _action_error_stats(pred, torch.zeros_like(pred), mask))
    assert (summed[0] / summed[1]).item() == pytest.approx(1.75)


def test_validation_all_masked_contributes_no_error_or_count():
    pred = torch.full((1, 2, 3), float("nan"))
    stats = _action_error_stats(pred, torch.zeros_like(pred), torch.zeros_like(pred))
    torch.testing.assert_close(stats, torch.zeros(2))


def test_validation_broadcast_mask_counts_expanded_elements():
    pred = torch.ones(2, 3, 4)
    mask = torch.tensor([[[1]], [[0]]])
    torch.testing.assert_close(
        _action_error_stats(pred, torch.zeros_like(pred), mask), torch.tensor([12.0, 12.0])
    )


def test_validation_accumulates_low_precision_inputs_in_float32():
    pred = torch.full((1, 2, 3), 300.0, dtype=torch.float16)
    stats = _action_error_stats(pred, torch.zeros_like(pred))
    torch.testing.assert_close(stats, torch.tensor([540000.0, 6.0]))


@pytest.mark.parametrize("all_masked", [False, True])
def test_main_reports_masked_validation_metric(monkeypatch, tmp_path, capsys, all_masked):
    class TinyPolicy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, batch, **kwargs):
            return self.weight.square()

        def sample_actions(self, batch, **kwargs):
            return batch["prediction"]

    samples = [
        {"actions": torch.zeros(1, 3), "prediction": torch.tensor([[2.0, 99.0, 99.0]]),
         "action_mask": torch.tensor([[1, 0, 0]])},
        {"actions": torch.zeros(1, 3), "prediction": torch.ones(1, 3),
         "action_mask": torch.ones(1, 3)},
    ]
    if all_masked:
        for sample in samples:
            sample["action_mask"].zero_()
    monkeypatch.setattr(train_loop, "_is_distributed", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_loop, "FakeActionDataset", lambda *args, **kwargs: samples)
    monkeypatch.setattr(train_loop, "collate_samples", torch.utils.data.default_collate)
    monkeypatch.setattr(train_loop, "DiTPolicy", TinyPolicy)
    config = TrainConfig(
        fake_data=True, pretrained_encoders=False, train_steps=1, batch_size=1,
        num_workers=0, val_every=1, val_batches=2, output_dir=str(tmp_path),
    )
    config.data.pin_memory = False
    train_loop.main(config)
    output = capsys.readouterr().out
    if all_masked:
        assert "val skipped (no valid action elements" in output
        assert "val_recon_error" not in output
    else:
        assert "val_recon_error 1.7500" in output


def test_main_rejects_empty_loader_before_initializing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(train_loop, "_is_distributed", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_loop, "FakeActionDataset", lambda *args, **kwargs: range(3))

    def unexpected_model(config):
        pytest.fail("Model must not be initialized for an empty training loader")

    monkeypatch.setattr(train_loop, "DiTPolicy", unexpected_model)
    config = TrainConfig(fake_data=True, batch_size=4, num_workers=0, output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Training loader has no batches"):
        train_loop.main(config)


@pytest.mark.parametrize("factor", [0, -1, 1.5, True, "2"])
def test_training_rejects_invalid_prefetch_before_initialization(monkeypatch, factor):
    config = TrainConfig(fake_data=True, num_workers=2)
    config.data.prefetch_factor = factor

    def unexpected_initialization(*args, **kwargs):
        pytest.fail("training runtime must not initialize for invalid loader settings")

    monkeypatch.setattr(torch, "set_float32_matmul_precision", unexpected_initialization)
    with pytest.raises(ValueError, match="prefetch_factor"):
        train_loop.main(config)
