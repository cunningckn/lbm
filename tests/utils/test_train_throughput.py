import pytest
from benchmarks.train_throughput import parse_args


def test_throughput_args_defaults():
    args = parse_args(["--batch-sizes", "1,4", "--json"])
    assert args.batch_sizes == "1,4"
    assert args.json is True
    assert args.warmup == 3


@pytest.mark.parametrize("argv", [["--iters", "0"], ["--warmup", "-1"], ["--batch-sizes", "0"]])
def test_throughput_args_reject_invalid_values(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_backward_only_leaves_parameters_unchanged_and_clears_gradients():
    import torch
    from benchmarks.train_throughput import _step

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2.)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = torch.ones(1, 1)
    _step(model, optimizer, batch, optimizer_step=False)
    _step(model, optimizer, batch, optimizer_step=False)
    assert model.weight.item() == 2.
    assert model.weight.grad.item() == 1.
    _step(model, optimizer, batch)
    assert model.weight.item() == pytest.approx(1.9)


def test_backward_only_cli_is_explicit():
    assert parse_args([]).optimizer_step
    assert not parse_args(["--no-optimizer-step"]).optimizer_step


@pytest.mark.parametrize("size", [0, -16, 15])
def test_rejects_unsupported_image_size(size):
    with pytest.raises(SystemExit):
        parse_args(["--image-size", str(size)])


def test_measurement_applies_image_size_to_every_input(monkeypatch):
    import torch
    from benchmarks import train_throughput as bench

    sizes = []

    def batch(config, batch_size, *, device, dtype, image_size):
        sizes.append(image_size)
        return torch.ones(batch_size, 1, device=device, dtype=dtype)

    class ScalarLoss(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, x):
            return (x * self.weight).sum()

    monkeypatch.setattr(bench, "DiTPolicy", ScalarLoss)
    monkeypatch.setattr(bench, "make_fake_batch", batch)
    result = bench.measure_batch(None, batch_size=1, device=torch.device("cpu"),
                                 dtype=torch.float32, warmup=1, iters=2, compile=False,
                                 optimizer_step=False, image_size=32)
    assert sizes == [32] * 4
    assert result["image_size"] == 32
    assert result["optimizer_step"] is False
