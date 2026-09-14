import argparse
import json
from pathlib import Path

import torch

from lbm.config import TrainConfig
from lbm.dataloader.mixture import load_dataset
from lbm.models.clip import CLIPTextEmbedder
from lbm.training_features import FeatureDataset
from lbm.training_sampler import CursorBatchSampler

parser = argparse.ArgumentParser(description="Compare matched live/cached CLIP vectors without publishing them")
parser.add_argument('--cache', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
cfg = TrainConfig(seed=123, batch_size=64, num_workers=0)
cfg.data.dataset = "agibot"
cfg.data.rescan = True
cfg.data.max_episodes = 2
cfg.data.instruction_mode = "subtask"
cfg.data.mmap_prebuild = False
cfg.model.action_length = 1.0
cfg.model.history_length = 0.3
cfg.model.history_freq = 10.0
cfg.model.history_time_encoding = True
cfg.model.state_history_length = 0.3
source = load_dataset(cfg)
source.set_mmap_allow_build(False)
cached = FeatureDataset(args.cache)
embedder = CLIPTextEmbedder(cfg.clip, device="cpu")
rows = []
for n, keys in enumerate(CursorBatchSampler(source, 64, seed=123)):
    if n == 3:
        break
    indices = [key[2] for key in keys]
    langs = [source[i]["lang"] for i in indices]
    live = embedder.encode(langs).bfloat16().float()
    saved = torch.stack([cached[i]["task_vec_clip"].float() for i in indices])
    rows.append(
        dict(
            batch=n,
            max_abs=(live - saved).abs().max().item(),
            relative_l2=((live - saved).norm() / live.norm()).item(),
            cosine_min=torch.nn.functional.cosine_similarity(live, saved).min().item(),
        )
    )
print(json.dumps(rows))
Path(args.output).write_text(json.dumps(rows, indent=2))
