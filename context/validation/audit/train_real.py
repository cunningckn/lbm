"""Real cached mixed-data training, finite-update checks and multiworker recovery."""
import argparse
import json
import time
from pathlib import Path

import torch

from lbm import train_loop
from lbm.config import TrainConfig

parser = argparse.ArgumentParser()
parser.add_argument('--train-cache', required=True)
parser.add_argument('--val-cache', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--steps', type=int, default=500)
args = parser.parse_args()
cfg = TrainConfig(feature_cache=args.train_cache, output_dir=args.output, batch_size=64,
                  num_workers=2, train_steps=args.steps, ckpt_every=args.steps, val_every=100,
                  val_batches=4, log_every=50, dump_batch=False)
cfg.data.val_dataset = args.val_cache
original_clip = torch.nn.utils.clip_grad_norm_
original_model = train_loop.DiTPolicy
losses, memory = [], []


def checked_clip(*a, **kw):
    return original_clip(*a, **kw, error_if_nonfinite=True)


def make_model(config):
    model = original_model(config)
    def record(_m, _inputs, loss):
        assert torch.isfinite(loss).all(), 'nonfinite loss'
        losses.append(float(loss.detach()))
        memory.append(torch.cuda.memory_allocated())
    model.register_forward_hook(record)
    return model


torch.nn.utils.clip_grad_norm_ = checked_clip
train_loop.DiTPolicy = make_model
start = time.perf_counter()
train_loop.main(cfg)
train_seconds = time.perf_counter() - start
cfg.resume = str(Path(args.output) / 'last.pt')
cfg.train_steps = args.steps + 10
start = time.perf_counter()
train_loop.main(cfg)
resume_seconds = time.perf_counter() - start
assert len(losses) == args.steps + 10
result = dict(updates=len(losses), train_seconds=train_seconds, resume_ten_updates_seconds=resume_seconds,
              first_loss=losses[0], last_loss=losses[-1],
              allocated_mib_after_warmup=[min(memory[20:args.steps])/2**20, max(memory[20:args.steps])/2**20])
(Path(args.output) / 'audit.json').write_text(json.dumps(result, indent=2))
print('AUDIT ' + json.dumps(result), flush=True)
