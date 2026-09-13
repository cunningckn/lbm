"""Short full-size policy validation on real subtask inputs; no bulk or image-cache build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lbm import train_loop
from lbm.config import TrainConfig
from lbm.dataloader import mixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('agibot', 'galaxea', 'mixed'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('this validation requires a real GPU')
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    cfg = TrainConfig(output_dir=str(output), seed=123, batch_size=16, num_workers=2,
                      train_steps=args.steps, val_every=5, val_batches=2, ckpt_every=10,
                      log_every=5, dump_batch=False, resume=args.resume)
    cfg.data.instruction_mode = 'subtask'
    cfg.data.rescan = True
    cfg.data.max_episodes = 4
    cfg.data.val_fraction = .25
    cfg.data.mmap_prebuild = False
    cfg.model.action_length = 1.
    cfg.model.action_freq = 15.
    cfg.optim.lr_warmup_steps = 10
    if args.dataset == 'mixed':
        cfg.data.data_mix = 'agibot,galaxea'
    else:
        cfg.data.dataset = args.dataset
    original_loader = mixture.load_dataset

    def load(*a, **kw):
        dataset = original_loader(*a, **kw)
        dataset.set_mmap_allow_build(False)
        return dataset

    mixture.load_dataset = load
    original_policy = train_loop.DiTPolicy

    class CheckedPolicy(original_policy):
        def forward(self, *a, **kw):
            loss = super().forward(*a, **kw)
            if not torch.isfinite(loss).all():
                raise ValueError('nonfinite real subtask training loss')
            return loss

    train_loop.DiTPolicy = CheckedPolicy
    train_loop.main(cfg)
    result = dict(dataset=args.dataset, completed_steps=args.steps, resumed_from=bool(args.resume),
                  batch=cfg.batch_size, max_episodes_per_source=4, val_fraction=.25,
                  action_length=1., requested_action_freq=15., peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  scope='short real-data functional validation, not convergence or matched throughput evidence')
    (output/f'completed-{args.steps}.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
