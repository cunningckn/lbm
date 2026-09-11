import random

import numpy as np
import torch

from lbm.train_loop import _load_checkpoint, _save_checkpoint


def test_checkpoint_restores_optimizer_scheduler_step_epoch_and_rng(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5 ** step)
    model(torch.ones(1, 3)).sum().backward()
    opt.step()
    scheduler.step()
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    random.random()
    np.random.rand()
    torch.rand(1)
    expected_after = (random.random(), np.random.rand(), torch.rand(1))
    _save_checkpoint(tmp_path / "state.pt", model, opt, scheduler, 7, 3, fsdp=False)
    random.random()
    np.random.rand()
    torch.rand(1)
    restored = torch.nn.Linear(3, 2)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda step: 0.5 ** step)
    assert _load_checkpoint(tmp_path / "state.pt", restored, opt2, sched2) == (7, 3)
    assert random.random() == expected_after[0]
    assert np.random.rand() == expected_after[1]
    torch.testing.assert_close(torch.rand(1), expected_after[2])
    assert sched2.last_epoch == scheduler.last_epoch
