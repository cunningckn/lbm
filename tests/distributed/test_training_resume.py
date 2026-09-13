"""Compare actual distributed optimizer updates across interruption, with worker prefetch."""
import os
from pathlib import Path

import pytest
import torch

from tests.distributed.test_fsdp import _free_port


def _resume_worker(rank, world, port, root, fsdp, stage, cached):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    from lbm import train_loop
    from lbm.config import TrainConfig
    from tests.helpers import tiny_dit_config

    original = train_loop.DiTPolicy
    losses = []
    def build(config):
        model = original(config)
        if cached:
            model.load_state_dict(torch.load(Path(root) / 'initial.pt', weights_only=True))
        model.register_forward_hook(lambda _m, _a, output:
                                    losses.append(float(output.detach())) if output.ndim == 0 else None)
        return model
    train_loop.DiTPolicy = build
    cfg = TrainConfig(model=tiny_dit_config(), fake_data=True, pretrained_encoders=False,
                      batch_size=2, num_workers=2, train_steps=4, val_every=2, val_batches=1,
                      ckpt_every=2, log_every=2, dump_batch=False, fsdp=fsdp, output_dir=root + '/full')
    if cached:
        cfg.fake_data = False
        cfg.feature_cache = root + '/cache-train'
        cfg.data.val_dataset = root + '/cache-val'
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
@pytest.mark.parametrize('fsdp,cached', [(False, False), (True, False), (True, True)])
def test_distributed_worker_resume_matches_updates(tmp_path, fsdp, cached):
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices')
    if fsdp:
        pytest.importorskip('megatron_fsdp')
    if cached:
        from dataclasses import asdict

        from lbm.models.dit import DiTPolicy
        from lbm.training_features import MODEL_FIELDS, backbone_fingerprint, write_feature_cache
        from tests.config.test_training_features import _cache_batch
        from tests.helpers import tiny_dit_config

        config = tiny_dit_config()
        model = DiTPolicy(config)
        batch = _cache_batch(config, model)
        metadata = dict(model={key: asdict(config)[key] for key in MODEL_FIELDS},
                        backbone_sha256=backbone_fingerprint(model.img_backbone.to(dtype=torch.bfloat16)))
        torch.save(model.state_dict(), tmp_path / 'initial.pt')
        for split in ('train', 'val'):
            write_feature_cache(tmp_path / f'cache-{split}', [batch] * 8, rows=16,
                                metadata=dict(metadata, episode_ids=[split]))
    # A killed earlier writer must not prevent retrying the same checkpoint step.
    (tmp_path / 'full' / '2.incomplete').mkdir(parents=True)
    for stage in ('full', 'cut', 'resume'):
        torch.multiprocessing.spawn(_resume_worker,
                                    args=(2, _free_port(), str(tmp_path), fsdp, stage, cached), nprocs=2, join=True)
    for rank in range(2):
        full = torch.load(tmp_path / f'full-{rank}.pt')
        split = torch.load(tmp_path / f'cut-{rank}.pt') + torch.load(tmp_path / f'resume-{rank}.pt')
        torch.testing.assert_close(torch.tensor(split), torch.tensor(full), rtol=0, atol=0)
