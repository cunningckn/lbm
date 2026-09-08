import torch

from lbm.models.dit import DiTPolicy
from lbm.utils.fake_data import make_fake_batch
from tests.helpers import tiny_dit_config


def test_padded_action_dims_do_not_affect_loss():
    cfg = tiny_dit_config(action_dim=8, state_dim=8, camera_keys=("cam",))
    model = DiTPolicy(cfg).eval()
    batch = make_fake_batch(cfg, batch_size=2)
    batch["actions"] = torch.zeros(2, cfg.chunk_length, 8)
    batch["actions"][..., :4] = 1.0
    mask = torch.zeros(2, cfg.chunk_length, 8, dtype=torch.bool)
    mask[..., :4] = True
    noise = torch.zeros_like(batch["actions"])
    noise[..., 4:] = 100.0
    t = torch.ones(2, 1, 1)
    with torch.no_grad():
        loss_masked = model({**batch, "action_mask": mask}, noise=noise.clone(), t=t.clone())
        loss_full = model(batch, noise=noise.clone(), t=t.clone())
    assert torch.isfinite(loss_masked)
    assert float(loss_masked) < float(loss_full)
