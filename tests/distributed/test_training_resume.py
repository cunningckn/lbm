"""Compare actual distributed optimizer updates across interruption, with worker prefetch."""
import os
from pathlib import Path

import pytest
import torch

from tests.distributed.test_fsdp import _free_port


def _resume_worker(rank, world, port, root, fsdp, stage):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    from lbm import train_loop
    from lbm.config import TrainConfig
    from tests.helpers import tiny_dit_config

    original = train_loop.DiTPolicy
    losses = []
    def build(config):
        model = original(config)
        model.register_forward_hook(lambda _m, _a, output:
                                    losses.append(float(output.detach())) if output.ndim == 0 else None)
        return model
    train_loop.DiTPolicy = build
    cfg = TrainConfig(model=tiny_dit_config(), fake_data=True, pretrained_encoders=False,
                      batch_size=2, num_workers=2, train_steps=4, val_every=2, val_batches=1,
                      ckpt_every=2, log_every=2, dump_batch=False, fsdp=fsdp, output_dir=root + '/full')
    cfg.optim.learning_rate = 1e-3
    cfg.optim.lr_warmup_steps = 1
    if stage != 'full':
        cfg.output_dir = root + '/split'
        cfg.train_steps = 2 if stage == 'cut' else 4
        if stage == 'resume':
            cfg.resume = root + '/split/2'
    train_loop.main(cfg)
    torch.save(losses, Path(root) / f'{stage}-{rank}.pt')


@pytest.mark.gpu
@pytest.mark.parametrize('fsdp', [False, True])
def test_distributed_worker_resume_matches_updates(tmp_path, fsdp):
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    if fsdp:
        pytest.importorskip('megatron_fsdp')
    # A killed earlier writer must not prevent retrying the same checkpoint step.
    (tmp_path / 'full' / '2.incomplete').mkdir(parents=True)
    for stage in ('full', 'cut', 'resume'):
        torch.multiprocessing.spawn(_resume_worker,
                                    args=(2, _free_port(), str(tmp_path), fsdp, stage), nprocs=2, join=True)
    for rank in range(2):
        full = torch.load(tmp_path / f'full-{rank}.pt')
        split = torch.load(tmp_path / f'cut-{rank}.pt') + torch.load(tmp_path / f'resume-{rank}.pt')
        torch.testing.assert_close(torch.tensor(split), torch.tensor(full), rtol=0, atol=0)
