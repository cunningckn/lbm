from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lbm.batch import policy_batch_from_loader
from lbm.dataloader.custom import CUSTOM_SPECS, CustomSingleDataset, Episode
from lbm.dataloader.pad import collate_fn


def test_timed_visual_and_state_history_survive_collation_and_state_masking():
    spec = CUSTOM_SPECS["libero"]
    state = np.repeat(np.arange(12, dtype=np.float32)[:, None], spec.state_dim, axis=1)
    images = {
        cam: np.broadcast_to(np.arange(12, dtype=np.uint8)[:, None, None, None], (12, 2, 2, 3)).copy()
        for cam in spec.camera_keys
    }
    episode = Episode(images, state, np.zeros((12, spec.action_dim), np.float32), "task")
    ds = CustomSingleDataset(
        spec,
        [episode],
        action_mode="abs",
        image_size=2,
        history_length=0.4,
        history_freq=10,
        history_time_encoding=True,
        state_history_length=0.4,
        state_history_freq=10,
    )
    sample = ds[1]
    assert sample["state_history"].shape == (4, spec.state_dim)
    assert sample["state_history_mask"].tolist() == [False, False, True, True]
    assert np.all(sample["state_history"][-1] == 1)
    assert np.all(sample["state"] == 1)
    np.testing.assert_array_equal(sample["image"][0][:, 0, 0, 0], sample["state_history"][:, 0])
    raw = collate_fn([sample, ds[5]])
    embedder = SimpleNamespace(encode=lambda texts: torch.zeros(len(texts), 512))
    kw = dict(
        camera_keys=spec.camera_keys,
        device=torch.device("cpu"),
        dtype=torch.float32,
        embedder=embedder,
        state_dim=spec.state_dim + 2,
    )
    batch = policy_batch_from_loader(raw, train=False, **kw)
    assert batch["state_history"].shape == (2, 4, spec.state_dim + 2)
    torch.testing.assert_close(batch["state_history"][:, -1], batch["state"])
    assert not batch["state_history"][..., -2:].any()
    hidden = policy_batch_from_loader(raw, train=True, mask_state_ratio=1.0, **kw)
    assert not hidden["state"].any() and not hidden["state_history"].any()
    assert not hidden["state_history_mask"].any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_time_encoding": True, "history_length": 10.0, "history_freq": 10.0},
        {"state_history_length": 10.0, "state_history_freq": 10.0},
    ],
)
def test_direct_dataset_constructor_enforces_history_capacity(kwargs):
    spec = CUSTOM_SPECS["libero"]
    episode = Episode({}, np.zeros((1, spec.state_dim)), np.zeros((1, spec.action_dim)), "task")
    with pytest.raises(ValueError, match="at most 64"):
        CustomSingleDataset(spec, [episode], action_mode="abs", **kwargs)
