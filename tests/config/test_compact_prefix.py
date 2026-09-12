import pytest
import torch

from lbm.models.dit import DiTPolicy
from lbm.utils.fake_data import make_fake_batch
from tests.helpers import tiny_dit_config


@pytest.mark.parametrize('device,dtype', [
    ('cpu', torch.float32), pytest.param('cuda', torch.bfloat16, marks=pytest.mark.gpu),
])
@pytest.mark.parametrize('prefix,probability', [(0, 1.), (1, 1.), (4, 0.), (4, .5), (4, 1.)])
def test_compact_prefix_matches_dense_loss_and_gradients(prefix, probability, device, dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    torch.manual_seed(10)
    cfg = tiny_dit_config()
    model = DiTPolicy(cfg)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=.06)
    model = model.to(device=device, dtype=dtype)
    batch = make_fake_batch(cfg, 4, device=device, image_size=32)
    for key, value in list(batch.items()):
        if isinstance(value, dict):
            batch[key] = {k: v.to(dtype=dtype) for k, v in value.items()}
        elif value.is_floating_point():
            batch[key] = value.to(dtype=dtype)
    batch['state_is_masked'] = torch.tensor([True, False, False, False])
    batch['action_mask'] = torch.ones_like(batch['actions'])
    batch['action_mask'][:, :, -2:] = 0
    noise = torch.randn_like(batch['actions'])
    t = torch.rand(4, 1, 1, device=device, dtype=dtype)
    kwargs = dict(noise=noise, t=t, max_action_prefix=prefix,
                  prefix_conditioning_prob=probability, prefix_noise_scale=.1)
    torch.manual_seed(21)
    dense = model(batch, **kwargs, compact_prefix_conditioning=False)
    dense.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(21)
    compact = model(batch, **kwargs)
    compact.backward()
    torch.testing.assert_close(compact, dense, rtol=.02 if device == 'cuda' else 2e-5, atol=2e-6)
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], rtol=.05 if device == 'cuda' else 2e-4,
                                       atol=2e-4 if device == 'cuda' else 2e-6)
