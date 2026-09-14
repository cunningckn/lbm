"""Bounded historical-policy train/resume smoke on LIBERO task 0, not a quality claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lbm import train_loop
from lbm.config import TrainConfig
from lbm.dataloader import mixture
from lbm.dataloader.custom.dataset import CustomMixtureDataset
from lbm.training_split import episode_view

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--resume', default='')
args = parser.parse_args()
output = Path(args.output)
if output.exists() and not args.resume:
    raise FileExistsError(output)
cfg = TrainConfig(output_dir=str(output), train_steps=args.steps, batch_size=32, num_workers=2,
                  log_every=10, val_every=50, val_batches=2, ckpt_every=50, dump_batch=False, resume=args.resume)
cfg.data.dataset = 'libero'
cfg.data.rescan = True
cfg.data.val_fraction = .2
cfg.data.mmap_prebuild = False
cfg.model.action_length = 5.
cfg.model.history_length = .3
cfg.model.history_freq = 10.
cfg.model.history_time_encoding = True
cfg.model.state_history_length = .3
cfg.model.state_history_freq = 10.
cfg.optim.lr_warmup_steps = 10
original = mixture.load_dataset


def load(*a, **kw):
    ds = original(*a, **kw).datasets[0]
    task = 'pick up the black bowl between the plate and the ramekin and place it on the plate'
    indices = [i for i, r in enumerate(ds.records) if r.lang.strip() == task]
    if not 2 <= len(indices) <= 100:
        raise ValueError(f'Unexpected task subset size: {len(indices)}')
    selected = episode_view(ds, indices)
    selected.set_mmap_allow_build(False)
    return CustomMixtureDataset([(selected, 1.)], seed=cfg.seed)


mixture.load_dataset = load
original_clip = torch.nn.utils.clip_grad_norm_


def clip(*a, **kw):
    kw['error_if_nonfinite'] = True
    return original_clip(*a, **kw)


torch.nn.utils.clip_grad_norm_ = clip
train_loop.main(cfg)
(output / f'completed-{args.steps}.json').write_text(json.dumps(dict(
    steps=args.steps, resumed=bool(args.resume), peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    scope='history train/resume/deploy functionality; not convergence')))
