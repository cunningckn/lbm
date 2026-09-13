import pytest
import torch

from lbm.models.dit import DiTPolicy
from lbm.utils.fake_data import make_fake_batch
from tests.helpers import tiny_dit_config


def _compare_prefix(prefix, probability, device, dtype, split, *, reference_compact=False, compiled=False):
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
    dense = model(batch, **kwargs, compact_prefix_conditioning=reference_compact,
                  split_prefix_conditioning=False)
    dense.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(21)
    original_keys = tuple(model.state_dict())
    if compiled:
        model.configure_conditioning(compile=True)
    compact = model(batch, **kwargs, split_prefix_conditioning=split)
    assert tuple(model.state_dict()) == original_keys
    compact.backward()
    torch.testing.assert_close(compact, dense, rtol=.02 if device == 'cuda' else 2e-5, atol=2e-6)
    for name, p in model.named_parameters():
        if name in gradients:
            if compiled and dtype == torch.bfloat16:
                # Fused backward reductions accumulate before casting to BF16.
                # Bound each tensor's relative L2 error by two BF16 epsilons;
                # elementwise relative error is unstable near zero gradients.
                expected, actual = gradients[name].float(), p.grad.float()
                error = (actual - expected).norm()
                assert error <= 2 * torch.finfo(dtype).eps * expected.norm() + 1e-8, name
                continue
            torch.testing.assert_close(p.grad, gradients[name], rtol=.05 if dtype == torch.bfloat16 else 2e-4,
                                       atol=2e-4 if dtype == torch.bfloat16 else 2e-6)


@pytest.mark.parametrize('device,dtype', [
    ('cpu', torch.float32), pytest.param('cuda', torch.bfloat16, marks=pytest.mark.gpu),
])
@pytest.mark.parametrize('prefix,probability', [(0, 1.), (1, 1.), (4, 0.), (4, .5), (4, 1.)])
@pytest.mark.parametrize('split', [False, True])
def test_compact_prefix_matches_dense_loss_and_gradients(prefix, probability, device, dtype, split):
    _compare_prefix(prefix, probability, device, dtype, split)


@pytest.mark.parametrize('device,dtype', [
    ('cpu', torch.float32), pytest.param('cuda', torch.bfloat16, marks=pytest.mark.gpu),
])
def test_split_prefix_covers_entire_chunk(device, dtype):
    # Compare to the existing compact projection: dense BF16 GEMMs with a
    # different row count already round differently, independently of splitting.
    _compare_prefix(20, 1., device, dtype, True, reference_compact=True)


@pytest.mark.gpu
@pytest.mark.parametrize('prefix,dtype', [(0, torch.bfloat16), (4, torch.bfloat16), (4, torch.float32)])
def test_compiled_conditioning_matches_eager_gradients(prefix, dtype):
    _compare_prefix(prefix, 1., 'cuda', dtype, True, reference_compact=True, compiled=True)
