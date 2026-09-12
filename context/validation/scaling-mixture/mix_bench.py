import argparse
import json

import torch

from lbm.config import TrainConfig
from lbm.train_loop import main

p = argparse.ArgumentParser()
p.add_argument('--workers', type=int, default=2)
p.add_argument('--steps', type=int, default=40)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--mmap', action='store_true')
a = p.parse_args()
cfg = TrainConfig(batch_size=a.batch, num_workers=a.workers, train_steps=a.steps,
                  log_every=10, val_every=1000, ckpt_every=1000, dump_batch=False,
                  output_dir=f'/tmp/lbm-mix-w{a.workers}-mmap{a.mmap}')
cfg.data.data_root_dir = '/mnt/kpfs/workspace/jinaoqun/Projects/lbm/datasets'
cfg.data.data_mix = 'kai0,agibot'
cfg.data.max_episodes = 1
cfg.data.use_mmap = a.mmap
cfg.data.use_mmap_frames = a.mmap
cfg.data.mmap_prebuild = False
main(cfg)
torch.cuda.synchronize()
print(json.dumps({'workers': a.workers, 'steps': a.steps, 'batch_size': a.batch,
                  'mmap': a.mmap, 'peak_memory_mib': torch.cuda.max_memory_allocated()/2**20}))
