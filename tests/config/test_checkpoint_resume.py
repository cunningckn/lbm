"""Exercise the next optimizer update and the real training-loop resume path."""

import copy
import random

import numpy as np
import pytest
import torch

from lbm import train_loop
from lbm.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from lbm.config import TrainConfig
from lbm.train_cli import build_train_config, parse_args


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_restores_next_update_and_rng(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    def components():
        model = torch.nn.Linear(3, 2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)
        return model, optimizer, scheduler

    def update(model, opt, scheduler):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(2, 3, device=device) * (random.random() + np.random.rand())
        model(x).square().mean().backward()
        opt.step()
        scheduler.step()

    model, opt, scheduler = components()
    update(model, opt, scheduler)
    signature = {"batches_per_epoch": 2}
    save_checkpoint(tmp_path / "state.pt", model, opt, scheduler, step=1, epoch=0,
                    batch_in_epoch=1, epoch_rng=capture_rng_state(), signature=signature)
    # The expectation must be drawn AFTER saving, not before the saved RNG state.
    expected = (random.random(), np.random.rand(), torch.rand(1, device=device))
    update(model, opt, scheduler)
    restored, opt2, sched2 = components()
    payload = load_checkpoint(tmp_path / "state.pt", restored, opt2, sched2, signature=signature)
    restore_rng_state(payload["rng_state"])
    assert_tree_equal((random.random(), np.random.rand(), torch.rand(1, device=device)), expected)
    update(restored, opt2, sched2)
    assert_tree_equal(restored.state_dict(), model.state_dict())
    assert_tree_equal(opt2.state_dict(), opt.state_dict())
    assert_tree_equal(sched2.state_dict(), scheduler.state_dict())


@pytest.mark.parametrize("cut", [1, 3, 4])
def test_main_resume_matches_uninterrupted_across_epochs(tmp_path, monkeypatch, cut):
    seen = []

    class Samples(torch.utils.data.Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, index):
            jitter = random.random() + np.random.rand() + torch.rand(()).item()
            return {"state": torch.tensor([index + jitter]), "actions": torch.tensor([[index * 0.01]])}

    class TinyPolicy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(()))

        def forward(self, batch, **kwargs):
            seen.append(batch["state"].clone())
            return (self.weight * batch["state"]).square().mean() * torch.rand(())

        def sample_actions(self, batch, **kwargs):
            return self.weight * batch["state"][:, None, :] + torch.rand(())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_loop, "_is_distributed", lambda: False)
    monkeypatch.setattr(train_loop, "DiTPolicy", TinyPolicy)
    monkeypatch.setattr(train_loop, "FakeActionDataset", lambda *args, **kwargs: Samples())
    monkeypatch.setattr(train_loop, "collate_samples", torch.utils.data.default_collate)

    def no_download(*args, **kwargs):
        pytest.fail("resume must not load/download pretrained encoders")

    monkeypatch.setattr(train_loop, "load_encoder_weights", no_download)
    config = TrainConfig(fake_data=True, pretrained_encoders=False, batch_size=2, num_workers=0,
                         bf16=False, train_steps=8, ckpt_every=1, val_every=2, val_batches=1,
                         log_every=1, output_dir=str(tmp_path / "full"))
    config.data.pin_memory = False
    train_loop.main(config)
    full_seen = list(seen)
    seen.clear()
    split = copy.deepcopy(config)
    split.train_steps = cut
    split.output_dir = str(tmp_path / "split")
    train_loop.main(split)
    split.resume = str(tmp_path / "split" / "last.pt")
    split.train_steps = 8
    split.pretrained_encoders = True
    train_loop.main(split)
    assert_tree_equal(seen, full_seen)
    full = torch.load(tmp_path / "full" / "last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "split" / "last.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "rng_state", "epoch", "step", "batch_in_epoch"):
        assert_tree_equal(full[key], resumed[key])


def test_atomic_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / "state.pt"
    target.write_bytes(b"previous valid checkpoint")
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)

    def fail(payload, out):
        out.write(b"partial write")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(target, model, opt, sched, step=0, epoch=0, batch_in_epoch=0,
                        epoch_rng=capture_rng_state(), signature={})
    assert target.read_bytes() == b"previous valid checkpoint"
    assert not target.with_suffix(".pt.tmp").exists()


@pytest.mark.parametrize("corruption", ["legacy", "signature", "cursor"])
def test_invalid_checkpoint_rejected_before_model_mutation(tmp_path, corruption):
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    signature = {"batches_per_epoch": 2}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, opt, sched, step=1, epoch=0, batch_in_epoch=1,
                    epoch_rng=capture_rng_state(), signature=signature)
    payload = torch.load(path, weights_only=False)
    if corruption == "legacy":
        del payload["format_version"]
    elif corruption == "signature":
        payload["signature"]["batches_per_epoch"] = 3
    else:
        payload["batch_in_epoch"] = 3
    torch.save(payload, path)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        load_checkpoint(path, model, opt, sched, signature=signature)
    assert_tree_equal(before, model.state_dict())


@pytest.mark.parametrize("mode", ["ddp", "ckpt"])
def test_unsupported_resume_fails_before_initialization(tmp_path, monkeypatch, mode):
    path = tmp_path / "state.pt"
    path.touch()
    config = TrainConfig(resume=str(path), num_workers=0)
    config.fsdp = mode == "fsdp"
    config.num_workers = 2 if mode == "workers" else 0
    config.load_pretrained = "weights.pt" if mode == "ckpt" else ""
    monkeypatch.setattr(train_loop, "_is_distributed", lambda: mode == "ddp")
    monkeypatch.setattr(train_loop, "DiTPolicy", lambda *_: pytest.fail("must reject before model construction"))
    with pytest.raises(ValueError, match="resume"):
        train_loop.main(config)


def test_cli_rejects_conflicting_initializers_and_keeps_fake_checkpoint_interval():
    with pytest.raises(SystemExit):
        parse_args(["--ckpt", "weights.pt", "--resume", "state.pt"])
    config = build_train_config(parse_args(["--fake-data", "--steps", "4", "--ckpt-every", "2"]))
    assert config.ckpt_every == 2
    short = build_train_config(parse_args(["--fake-data", "--steps", "2", "--val-every", "10"]))
    long = build_train_config(parse_args(["--fake-data", "--steps", "8", "--val-every", "10"]))
    assert short.val_every == long.val_every == 10
