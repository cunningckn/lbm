#!/usr/bin/env python3
"""Prebuild frozen vision features using training CLI options; output-dir is a new cache directory."""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from torch.utils.data import DataLoader, Subset

from lbm.batch import infer_policy_io, policy_batch_from_loader
from lbm.dataloader.mixture import load_dataset
from lbm.dataloader.pad import collate_fn, dataloader_worker_init_fn
from lbm.feature_shards import write_sharded_cache
from lbm.models.clip import CLIPTextEmbedder
from lbm.models.dit import DiTPolicy
from lbm.models.encoders import build_vision_backbone, load_encoder_weights
from lbm.train_cli import build_train_config, parse_args
from lbm.training_features import MODEL_FIELDS, backbone_fingerprint
from lbm.training_split import dataset_fingerprint, episode_ids, prepare_validation
from lbm.utils.preprocess import _jsonable


def main(argv=None):
    args = parse_args(argv)
    cfg = build_train_config(args)
    if not 0 <= cfg.data.val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if cfg.fake_data or cfg.feature_cache or cfg.resume or cfg.load_pretrained:
        raise ValueError('prebuild requires real source data and pretrained encoder weights')
    if cfg.model.train_vision_encoder or cfg.model.language_encoder != 'none':
        raise ValueError('prebuild currently supports frozen vision with language_encoder=none')
    if Path(cfg.output_dir).exists():
        raise FileExistsError('choose a new --output-dir for the feature cache')
    dataset = load_dataset(cfg)
    if cfg.data.val_fraction:
        training, validation = prepare_validation(dataset, (), cfg)
        dataset = training if args.feature_split == 'train' else validation
    elif args.feature_split == 'val':
        raise ValueError('--feature-split val requires --val-fraction')
    io = infer_policy_io(dataset)
    cfg.model.camera_keys = io['camera_keys']
    cfg.model.state_dim = max(io['state_dim'], cfg.data.max_state_dim or 0)
    cfg.model.action_dim = max(io['action_dim'], cfg.data.max_action_dim or 0)
    cfg.model.action_freq = io['chunk_length'] / cfg.model.action_length
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if cfg.bf16 and device.type == 'cuda' else torch.float32
    # Avoid allocating the unused DiT during feature extraction.
    encoder = SimpleNamespace(config=cfg.model, camera_keys=cfg.model.camera_keys,
                              img_backbone=build_vision_backbone(cfg.model))
    loaded = load_encoder_weights(encoder, download=False)
    if cfg.model.vision_encoder not in loaded:
        raise FileNotFoundError('matching pretrained vision weights are required')
    encoder.img_backbone.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if dtype == torch.bfloat16 and hasattr(encoder.img_backbone, 'set_bfloat16'):
        encoder.img_backbone.set_bfloat16(True)
    embedder = CLIPTextEmbedder(cfg.clip, device='cpu')
    norms = {ds.spec.name: _jsonable(ds.norm_stats) for ds in dataset.datasets}
    sources, episode_ranges, start = [], [], 0
    for source, weight in zip(dataset.datasets, dataset.weights, strict=True):
        sources.append(dict(name=source.spec.name, start=start, stop=start + len(source), weight=weight))
        previous = 0
        for stop in source._offsets:
            episode_ranges.append([start + previous, start + int(stop)])
            previous = int(stop)
        start += len(source)
    metadata = dict(sources=sources, episode_ranges=episode_ranges,
                    episode_ids=[key for ds in dataset.datasets for key in episode_ids(ds)],
                    source_fingerprint=dataset_fingerprint(dataset),
                    model={key: getattr(cfg.model, key) for key in MODEL_FIELDS},
                    backbone_sha256=backbone_fingerprint(encoder.img_backbone),
                    normalization=norms,
                    action_spaces={ds.spec.name: [sl.as_dict() for sl in ds._space] for ds in dataset.datasets},
                    source_config=asdict(cfg))

    def batches(start):
        loader = DataLoader(Subset(dataset, range(start, len(dataset))), batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
                            worker_init_fn=dataloader_worker_init_fn if cfg.num_workers else None)
        count = start
        with torch.no_grad():
            for raw in loader:
                batch = policy_batch_from_loader(raw, camera_keys=cfg.model.camera_keys,
                                                device=device, dtype=dtype, embedder=embedder,
                                                state_dim=cfg.model.state_dim, action_dim=cfg.model.action_dim,
                                                action_steps=cfg.model.chunk_length, train=False)
                batch['vision_features'] = DiTPolicy.encode_vision_features(encoder, batch.pop('images'))
                batch.setdefault('camera_mask', torch.ones(len(batch['state']), len(cfg.model.camera_keys),
                                                           dtype=torch.bool, device=device))
                batch.setdefault('action_mask', torch.ones_like(batch['actions'], dtype=torch.bool))
                count += len(batch['state'])
                if count % (cfg.batch_size * 20) == 0:
                    print(f'features {count}/{len(dataset)}', flush=True)
                yield batch

    write_sharded_cache(cfg.output_dir, batches, rows=len(dataset), metadata=metadata,
                        shard_rows=args.feature_shard_rows)
    print(f'feature cache ready: {cfg.output_dir} rows={len(dataset)}', flush=True)


if __name__ == '__main__':
    main()
