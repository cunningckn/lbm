import json

import torch

from lbm.config import TrainConfig
from lbm.train_loop import main

cfg = TrainConfig(batch_size=4, num_workers=2, train_steps=20, log_every=5,
                  val_every=1000, ckpt_every=1000, output_dir='/tmp/lbm-real-train-output',
                  dump_batch=False)
cfg.data.dataset = '/mnt/kpfs/workspace/jinaoqun/Projects/lbm/datasets/kai0'
cfg.data.robot_type = 'kai0'
cfg.data.max_episodes = 1
cfg.data.use_mmap = False
cfg.data.use_mmap_frames = False
cfg.data.mmap_prebuild = False
main(cfg)
torch.cuda.synchronize()
print(json.dumps({'peak_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'dataset': 'kai0', 'max_episodes': 1, 'steps': 20, 'batch_size': 4,
                  'num_workers': 2, 'mmap': False}))
