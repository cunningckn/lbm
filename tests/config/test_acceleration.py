import copy

import pytest
import torch

from lbm.config import OptimConfig, TrainConfig
from lbm.optim import build_adamw
from lbm.train_cli import build_train_config, parse_args
from lbm.train_loop import _resume_signature


def test_acceleration_cli_and_legacy_default_signature():
    cfg = build_train_config(parse_args(['--fake-data', '--compile-conditioning', '--fused-adamw']))
    assert cfg.compile_conditioning and cfg.fused_adamw
    assert cfg.data.prefetch_factor == TrainConfig().data.prefetch_factor
    tuned = build_train_config(parse_args(['--prefetch-factor', '1']))
    assert tuned.data.prefetch_factor == 1
    with pytest.raises(SystemExit):
        parse_args(['--compile', '--compile-conditioning'])
    class Loader:
        dataset = range(8)
        def __len__(self):
            return 2
    default = TrainConfig(fake_data=True)
    old = _resume_signature(default, Loader(), torch.device('cpu'))
    assert 'compile_conditioning' not in old and 'fused_adamw' not in old
    default.compile_conditioning = default.fused_adamw = True
    new = _resume_signature(default, Loader(), torch.device('cpu'))
    assert new.pop('compile_conditioning') and new.pop('fused_adamw')
    assert new == old


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_fused_adamw_update_error_and_exact_restore(dtype):
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    torch.manual_seed(123)
    initial = torch.nn.Linear(32, 16).cuda().to(dtype)
    models = [copy.deepcopy(initial) for _ in range(2)]
    optimizers = [build_adamw(model, OptimConfig(), fused=fused)
                  for model, fused in zip(models, [False, True])]
    gradients = [[torch.randn_like(p) for p in initial.parameters()] for _ in range(10)]

    def update(model, optimizer, grads):
        for p, gradient in zip(model.parameters(), grads):
            p.grad = gradient.clone()
        optimizer.step()

    for i, grads in enumerate(gradients):
        for model, optimizer in zip(models, optimizers):
            update(model, optimizer, grads)
        if i == 4:
            state = copy.deepcopy((models[1].state_dict(), optimizers[1].state_dict()))
    for eager, fused in zip(models[0].parameters(), models[1].parameters()):
        # FP32 fused multiply-add changes rounding over ten updates; bound
        # accumulation by one dtype epsilon per update (BF16 keeps two).
        rounding = (len(gradients) if dtype == torch.float32 else 2) * torch.finfo(dtype).eps
        tolerance = rounding * float(eager.abs().max())
        torch.testing.assert_close(fused, eager, rtol=rounding, atol=tolerance)
    restored = copy.deepcopy(initial)
    optimizer = build_adamw(restored, OptimConfig(), fused=True)
    restored.load_state_dict(state[0])
    optimizer.load_state_dict(state[1])
    for grads in gradients[5:]:
        update(restored, optimizer, grads)
    for expected, actual in zip(models[1].parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for p, q in zip(models[1].parameters(), restored.parameters()):
        for name, expected in optimizers[1].state[p].items():
            torch.testing.assert_close(optimizer.state[q][name], expected, rtol=0, atol=0)
