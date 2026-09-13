import numpy as np
import pytest
import torch
from tests.helpers import tiny_dit_config

from lbm.history import frame_history
from lbm.policy import LBMPolicy, _identity_stats


class CaptureModel:
    def sample_actions(self, batch, *, num_steps):
        self.batch = batch
        return torch.zeros(1, 1, 3)


@pytest.mark.parametrize("timed_vision", [False, True])
def test_online_history_matches_physical_frames_and_resets_context(timed_vision):
    cfg = tiny_dit_config(camera_keys=("image",), state_dim=3, action_dim=3)
    cfg.history_length = cfg.state_history_length = 0.4
    cfg.history_freq = cfg.state_history_freq = 10
    cfg.history_time_encoding = timed_vision
    model = CaptureModel()
    policy = LBMPolicy(
        model,
        config=cfg,
        device=torch.device("cpu"),
        norm_stats={key: _identity_stats(3) for key in ("state", "actions")},
    )
    policy._task_vec = lambda prompt: torch.zeros(1, cfg.task_embed_dim)
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    obs = dict(images={"image": pixels}, prompt="reach", episode_id="a")
    with pytest.raises(ValueError, match="timestamp"):
        policy.infer(dict(obs, state=np.zeros(3)))
    for i in range(7):
        pixels.fill(i)
        policy.infer(dict(obs, timestamp=1720000000.0 + i / 15, state=np.full(3, i / 10)))
    batch = model.batch
    selected = frame_history(6, lower=0, fps=15, length=0.4, frequency=10)
    torch.testing.assert_close(batch["state_history"][0, :, 0], torch.tensor(selected.indices / 10).float())
    if timed_vision:
        torch.testing.assert_close(batch["history_offsets"], batch["state_history_offsets"])
        assert batch["history_mask"].all()
    assert batch["images"]["image"].shape == (1, 4, 3, 224, 224)
    # Snapshot owns its data; later camera writes cannot alter buffered images.
    if timed_vision:
        snapshot = policy._observations.select(length=0.4, frequency=10)[0][0][0]["image"]
        saved = snapshot.clone()
    pixels.fill(255)
    if timed_vision:
        torch.testing.assert_close(snapshot, saved)
    policy.infer(dict(obs, timestamp=0.0, state=np.zeros(3), subtask_id="new"))
    assert model.batch["state_history_mask"].tolist() == [[False, False, False, True]]
    if timed_vision:
        assert model.batch["history_mask"].tolist() == [[False, False, False, True]]
    else:
        assert len(policy._history["image"]) == 1
    assert len(policy._observations) == 1
    assert policy.metadata()["requires_timestamp"]
    policy.reset()
    assert len(policy._observations) == 0 and policy._last_raw_state is None
