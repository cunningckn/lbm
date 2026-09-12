"""Server validation harness: disjoint episodes, training-only stats, fixed-noise validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import lbm.dataloader.mixture as mixture
import lbm.train_loop as runner
from lbm.config import TrainConfig
from lbm.dataloader.custom import CustomMixtureDataset
from lbm.dataloader.custom.dataset import CustomSingleDataset
from lbm.utils.preprocess import compute_norm_stats, save_norm_stats

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--steps', type=int, default=5000)
parser.add_argument('--batch-size', type=int, default=4)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
cfg = TrainConfig(batch_size=args.batch_size, num_workers=0, train_steps=args.steps,
                  log_every=100, val_every=500, val_batches=16, ckpt_every=args.steps,
                  dump_batch=False, output_dir=str(output))
cfg.data.data_mix = 'kai0,agibot'
cfg.data.data_root_dir = args.data_root
cfg.data.max_episodes = 20
cfg.data.val_dataset = 'experiment-heldout-episodes'
cfg.data.use_mmap = cfg.data.use_mmap_frames = cfg.data.mmap_prebuild = False
loaded = mixture.load_dataset(cfg)
train, validation = [], []
for source in loaded.datasets:
    records = list(source.records)
    assert len(records) >= 5
    order = np.random.default_rng(cfg.seed).permutation(len(records))
    cut = max(1, len(records) // 5)
    groups = [[records[i] for i in order[cut:]], [records[i] for i in order[:cut]]]
    # Both selected native sources use one vector file per episode.
    assert not {r.path for r in groups[0]} & {r.path for r in groups[1]}
    common = dict(action_mode=source.action_mode, action_length=source.action_length,
                  action_freq=source.action_freq, history_length=0, history_freq=10,
                  root=source.root, use_mmap=False)
    tr = CustomSingleDataset(source.spec, records=groups[0], **common)
    va = CustomSingleDataset(source.spec, records=groups[1], **common)
    payload = compute_norm_stats(tr, progress=False, scratch_dir='/tmp')
    tr.norm_stats = va.norm_stats = payload['norm_stats']
    save_norm_stats(output / f'{source.spec.name}-norm.json', payload)
    train.append((tr, 1.0))
    validation.append((va, 1.0))
    print('SPLIT', source.spec.name, len(groups[0]), len(groups[1]), len(tr), len(va), flush=True)
train_ds = CustomMixtureDataset(train)
val_ds = CustomMixtureDataset(validation, mode='val')
# Spread the fixed validation budget over all held-out episodes and both sources.
per_source = cfg.val_batches * cfg.batch_size // len(validation)
val_ds._map = [(d, int(i)) for row in zip(*[
    np.linspace(0, len(ds) - 1, per_source, dtype=int) for ds, _ in validation
], strict=True) for d, i in enumerate(row)]
mixture.load_dataset = lambda *a, mode='train', **kw: train_ds if mode == 'train' else val_ds
original_clip = torch.nn.utils.clip_grad_norm_


def checked_clip(*a, **kw):
    kw['error_if_nonfinite'] = True
    return original_clip(*a, **kw)


torch.nn.utils.clip_grad_norm_ = checked_clip
original_eval = runner.evaluate_actions
curve = []


def evaluate(*a, **kw):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(2026)
        stats = original_eval(*a, **kw)
    curve.append(float((stats[0] / stats[1]).item()))
    print('HELDOUT', len(curve) * cfg.val_every, curve[-1], flush=True)
    return stats


runner.evaluate_actions = evaluate
original_model = runner.DiTPolicy
observed = []


def make_model(config):
    model = original_model(config)
    def check(module, inputs, loss):
        assert torch.isfinite(loss).all(), 'nonfinite loss'
        observed.append(torch.cuda.memory_allocated() / 2**20)
    model.register_forward_hook(check)
    return model


runner.DiTPolicy = make_model
runner.main(cfg)
assert len(observed) == args.steps
cfg.resume = str(output / 'last.pt')
cfg.train_steps = args.steps + 10
runner.main(cfg)
assert len(observed) == args.steps + 10
summary = dict(updates=len(observed), heldout_curve=curve,
               allocated_mib_after_warmup=[min(observed[100:args.steps]), max(observed[100:args.steps])])
(output / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
