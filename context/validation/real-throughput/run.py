"""Measure complete real-data training updates, including loader wait and AdamW."""
import argparse
import json
import os
import time

import torch

import lbm.dataloader.mixture as mixture
import lbm.distributed as distributed
import lbm.train_loop as runner
from lbm.config import TrainConfig
from lbm.utils.preprocess import load_norm_stats

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['live', 'features'], required=True)
p.add_argument('--batch', type=int, required=True)
p.add_argument('--workers', type=int, default=2)
p.add_argument('--dense', action='store_true')
p.add_argument('--fsdp', action='store_true')
p.add_argument('--fsdp-group-size', type=int, default=1)
p.add_argument('--mmap', action='store_true')
p.add_argument('--baseline-batch-module', default='', help='optional original batch.py for CPU-image ablation')
p.add_argument('--cache', default='/tmp/lbm-real-features')
p.add_argument('--data-root', required=True)
p.add_argument('--norm-dir', default='/tmp/lbm-long-heldout')
p.add_argument('--output', required=True)
p.add_argument('--steps', type=int, default=70)
a = p.parse_args()
cfg = TrainConfig(batch_size=a.batch, num_workers=a.workers, train_steps=a.steps,
                  log_every=10, val_every=10000, ckpt_every=10000, dump_batch=False,
                  output_dir=a.output, fsdp=a.fsdp)
cfg.parallel.fsdp_group_size = a.fsdp_group_size
cfg.data.data_mix = 'kai0,agibot'
cfg.data.data_root_dir = a.data_root
cfg.data.max_episodes = 4
cfg.data.use_mmap = cfg.data.use_mmap_frames = a.mmap
cfg.data.mmap_prebuild = False
cfg.feature_cache = a.cache if a.mode == 'features' else ''
original_load = mixture.load_dataset

def load(*args, **kwargs):
    dataset = original_load(*args, **kwargs)
    for ds in dataset.datasets:
        ds.norm_stats = load_norm_stats(f'{a.norm_dir}/{ds.spec.name}-norm.json')
    return dataset
mixture.load_dataset = load
original_forward = runner.DiTPolicy.forward

def forward(self, *args, **kwargs):
    kwargs['compact_prefix_conditioning'] = not a.dense
    return original_forward(self, *args, **kwargs)
runner.DiTPolicy.forward = forward
if a.baseline_batch_module:
    import importlib.util
    spec = importlib.util.spec_from_file_location('baseline_batch', a.baseline_batch_module)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    runner.policy_batch_from_loader = baseline.policy_batch_from_loader
original_build = runner.build_adamw
original_wrap = distributed.wrap_policy_fsdp
ends = []

def instrument(optimizer):
    def end(*unused):
        torch.cuda.synchronize()
        ends.append(time.perf_counter())
        if len(ends) == 10:
            torch.cuda.reset_peak_memory_stats()
    optimizer.register_step_post_hook(end)
    return optimizer

def build(*args, **kwargs):
    optimizer = original_build(*args, **kwargs)
    return optimizer if a.fsdp else instrument(optimizer)

def wrap(*args, **kwargs):
    model, optimizer = original_wrap(*args, **kwargs)
    return model, instrument(optimizer)

runner.build_adamw = build
distributed.wrap_policy_fsdp = wrap
runner.main(cfg)
assert len(ends) == a.steps
elapsed = ends[-1] - ends[9]
result = dict(vars(a), rank=int(os.environ.get('RANK', '0')),
              world=int(os.environ.get('WORLD_SIZE', '1')), measured_updates=a.steps-10, seconds=elapsed,
              samples_per_second=(a.steps-10)*a.batch/elapsed,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
print('RESULT ' + json.dumps(result), flush=True)
