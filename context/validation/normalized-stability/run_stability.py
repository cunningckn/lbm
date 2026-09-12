import json
from pathlib import Path

import torch

import lbm.dataloader.mixture as mixture
import lbm.train_loop as runner
from lbm.config import TrainConfig
from lbm.utils.preprocess import compute_norm_stats, save_norm_stats

output = Path('/tmp/lbm-normalized-stability')
output.mkdir(exist_ok=True)
cfg = TrainConfig(batch_size=4, num_workers=0, train_steps=200, log_every=20,
                  val_every=1000, ckpt_every=200, dump_batch=False, output_dir=str(output))
cfg.data.data_mix = 'kai0,agibot'
cfg.data.data_root_dir = '/mnt/kpfs/workspace/jinaoqun/Projects/lbm/datasets'
cfg.data.max_episodes = 1
cfg.data.use_mmap = cfg.data.use_mmap_frames = cfg.data.mmap_prebuild = False
original_load = mixture.load_dataset
prepared = original_load(cfg)
stats = {}
for ds in prepared.datasets:
    payload = compute_norm_stats(ds, progress=False)
    stats[ds.spec.name] = payload['norm_stats']
    save_norm_stats(output / f'{ds.spec.name}-norm.json', payload)
    print('NORMALIZATION', ds.spec.name, payload['norm_stats']['actions']['count'], flush=True)
del prepared

def load_with_stats(*args, **kwargs):
    ds = original_load(*args, **kwargs)
    for single in ds.datasets:
        single.norm_stats = stats[single.spec.name]
    return ds

mixture.load_dataset = load_with_stats
original_model = runner.DiTPolicy
observed = []

def create_model(config):
    model = original_model(config)
    def check(module, args, loss):
        assert torch.isfinite(loss).all(), 'non-finite loss'
        observed.append(float(torch.cuda.memory_allocated()/2**20))
        if len(observed) % 50 == 0:
            print('MEMORY', len(observed), observed[-1], flush=True)
    model.register_forward_hook(check)
    return model

runner.DiTPolicy = create_model
runner.main(cfg)
assert len(observed) == 200
assert max(observed[50:200])-min(observed[50:200]) < 1024, 'allocated memory grew by >1 GiB'
cfg.resume = str(output / 'last.pt')
cfg.train_steps = 205
runner.main(cfg)
assert len(observed) == 205
print(json.dumps({'updates':len(observed), 'resume_from':200, 'resume_updates':5,
                  'allocated_range_mib_after_warmup': [min(observed[50:200]), max(observed[50:200])]}), flush=True)
# Remove only this experiment's large checkpoints; retain normalization/logs.
for name in ('last.pt', '200.pt'):
    (output / name).unlink()
