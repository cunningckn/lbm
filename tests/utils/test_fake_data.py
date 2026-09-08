import pytest
import torch
from torch.utils.data import DataLoader

from lbm.config import DiTConfig
from lbm.utils.fake_data import (
    FakeActionDataset,
    collate_samples,
    describe_batch,
    make_fake_batch,
    make_fake_sample,
    move_batch_to_device,
)


def test_make_fake_batch_t5_token_vocab():
    cfg = DiTConfig(language_encoder="t5", t5_vocab_size=100, language_max_length=8, task_embed_dim=512)
    batch = make_fake_batch(cfg, batch_size=3)
    assert batch["task_tokens"].dtype == torch.long
    assert batch["task_tokens"].shape == (3, 8)
    assert int(batch["task_tokens"].max()) < 100
    assert batch["task_token_mask"].shape == (3, 8)
    assert torch.equal(batch["task_token_mask"] > 0, batch["task_tokens"] != 0)


def test_make_fake_batch_shapes(config):
    batch = make_fake_batch(config, batch_size=3)
    assert batch["state"].shape == (3, config.state_dim)
    assert batch["actions"].shape == (3, config.chunk_length, config.action_dim)
    assert batch["task_vec_clip"].shape == (3, config.task_embed_dim)
    for cam in config.camera_keys:
        assert batch["images"][cam].shape == (3, 3, 224, 224)
    assert "state:" in describe_batch(batch)


def test_make_fake_batch_rejects_bad_image_size(config):
    with pytest.raises(ValueError, match="multiple of 16"):
        make_fake_batch(config, image_size=30)


def test_collate_and_move_device(config):
    samples = [make_fake_sample(config) for _ in range(2)]
    batch = collate_samples(samples)
    assert batch["state"].shape[0] == 2
    moved = move_batch_to_device(batch, "cpu")
    assert moved["state"].device.type == "cpu"
    assert set(moved["images"]) == set(config.camera_keys)


def test_fake_dataset_loader_trains_one_step(model, clip, config, device, dtype):
    ds = FakeActionDataset(config, length=2, seed=0)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_samples)
    batch = next(iter(loader))
    for key, value in batch.items():
        if key == "images":
            batch[key] = {cam: img.to(device=device, dtype=dtype) for cam, img in value.items()}
        elif key == "task_tokens":
            batch[key] = value.to(device=device)
        else:
            batch[key] = value.to(device=device, dtype=dtype)
    batch["task_vec_clip"] = clip.encode("put the bottles in the bin").to(
        device=device, dtype=dtype
    )
    model.train()
    loss = model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    model.zero_grad(set_to_none=True)
