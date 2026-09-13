#!/usr/bin/env python3
"""Matched single-GPU compute, cache and online-image training measurements.

Use a provenance-bearing feature cache as the common model/data contract.
Throughput and synchronized stage profiling are separate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lbm.batch import policy_batch_from_loader
from lbm.config import TrainConfig
from lbm.feature_shards import atomic_json
from lbm.models.clip import CLIPTextEmbedder
from lbm.models.dit import DiTPolicy
from lbm.models.encoders import load_encoder_weights
from lbm.optim import build_adamw, count_trainable
from lbm.training_features import FeatureDataset
from lbm.training_loader import make_loader
from lbm.training_sampler import CursorBatchSampler
from lbm.training_split import dataset_fingerprint, prepare_validation
from lbm.utils.fake_data import move_batch_to_device

MODES = ('synthetic-features', 'synthetic-images', 'resident-features', 'resident-images', 'features', 'live')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=MODES)
    p.add_argument('--cache', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--mmap-images', action='store_true',
                   help='online modes: read existing JPEG mmap caches; never build during measurement')
    p.add_argument('--prefetch-factor', type=int, default=None, help='override cache recipe loader prefetch')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--steps', type=int, default=60, help='measured optimizer updates, excluding warmup')
    p.add_argument('--profile-steps', type=int, default=10)
    p.add_argument('--fused-adamw', action='store_true', help='experimental fused optimizer comparison')
    p.add_argument('--attention', choices=('auto', 'sdpa'), default='auto')
    p.add_argument('--compile-conditioning', action='store_true', help='compile only modulation/gating functions')
    p.add_argument('--split-prefix', action='store_true', help='bound expanded conditions to the prefix window')
    p.add_argument('--kernels', action='store_true', help='separate one-step operator diagnostic')
    a = p.parse_args(argv)
    if min(a.batch, a.steps, a.profile_steps) <= 0 or min(a.workers, a.warmup) < 0:
        p.error('batch/steps/profile-steps must be positive; workers/warmup must be nonnegative')
    if a.prefetch_factor is not None and a.prefetch_factor <= 0:
        p.error('prefetch-factor must be positive')
    if a.mmap_images and a.mode not in ('live', 'resident-images'):
        p.error('mmap-images requires live or resident-images mode')
    return a


def matched_config(metadata, *, batch, workers, data_root):
    """Restore the cache producer's recipe; never silently use a smaller default model."""
    source = metadata['source_config']
    cfg = TrainConfig()
    for name in ('model', 'optim', 'flow', 'data', 'clip'):
        setattr(cfg, name, type(getattr(cfg, name))(**source[name]))
    cfg.model.camera_keys = tuple(cfg.model.camera_keys)
    cfg.seed = source['seed']
    cfg.bf16 = source['bf16']
    cfg.batch_size, cfg.num_workers = batch, workers
    cfg.data.data_root_dir = data_root
    cfg.data.mmap_prebuild = False
    if cfg.model.train_vision_encoder or cfg.model.language_encoder != 'none':
        raise ValueError('comparison requires a frozen vision encoder and language_encoder=none')
    return cfg


def contract(cfg):
    fields = dict(model=asdict(cfg.model), flow=asdict(cfg.flow), optim=asdict(cfg.optim),
                  bf16=cfg.bf16, seed=cfg.seed, batch=cfg.batch_size, image_size=224)
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return fields, digest


def tensor_tree(value, fn):
    if isinstance(value, dict):
        return {key: tensor_tree(item, fn) for key, item in value.items()}
    return fn(value) if torch.is_tensor(value) else value


def mask_state(batch, ratio):
    # All six modes use the same masking recipe without modifying a reused batch.
    result = dict(batch)
    if ratio > 0:
        masked = torch.rand(len(batch['state']), device=batch['state'].device) < ratio
        result['state_is_masked'] = masked
        result['state'] = batch['state'].masked_fill(masked[:, None], 0)
    return result


def batches_forever(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        loader.generator.manual_seed(sampler.seed + epoch)
        yield from loader
        epoch += 1


class StageTimer:
    def __init__(self):
        self.enabled = False
        self.track_loader = False
        self.loader_seconds = []
        self.seconds = defaultdict(float)

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            track = self.track_loader and name == 'loader_wait'
            start = time.perf_counter() if track else None
            yield
            if track:
                self.loader_seconds.append(time.perf_counter() - start)
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.seconds[name] += time.perf_counter() - start


def main(argv=None):
    a = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('matched_throughput requires a CUDA GPU')
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = FeatureDataset(a.cache)
    cfg = matched_config(cache.metadata, batch=a.batch, workers=a.workers, data_root=a.data_root)
    if a.prefetch_factor is not None:
        cfg.data.prefetch_factor = a.prefetch_factor
    if a.mmap_images:
        cfg.data.use_mmap = cfg.data.use_mmap_frames = True
    cache.apply_config(cfg)
    spec, digest = contract(cfg)
    print('CONTRACT ' + json.dumps(dict(sha256=digest, **spec)), flush=True)
    device = torch.device('cuda')
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    torch.set_float32_matmul_precision('high')
    from lbm.models.attention import set_use_flash_attn

    set_use_flash_attn(False if a.attention == 'sdpa' else None)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = DiTPolicy(cfg.model)
    if cfg.model.vision_encoder not in load_encoder_weights(model, download=False):
        raise FileNotFoundError('matching pretrained encoder weights are required')
    model.to(device=device, dtype=dtype).train()
    if cfg.bf16 and hasattr(model.img_backbone, 'set_bfloat16'):
        model.img_backbone.set_bfloat16(True)
    cache.check_backbone(model)
    model.configure_conditioning(compile=a.compile_conditioning)
    optimizer = build_adamw(model, cfg.optim, fused=True if a.fused_adamw else None)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / max(cfg.optim.lr_warmup_steps, 1), 1.0))

    online = a.mode in ('live', 'resident-images')
    loader = sampler = stream = embedder = resident = None
    if online:
        from lbm.dataloader.mixture import load_dataset
        from lbm.dataloader.pad import collate_fn

        dataset, _ = prepare_validation(load_dataset(cfg), (), cfg)
        if len(dataset) != len(cache) or dataset_fingerprint(dataset) != cache.metadata['source_fingerprint']:
            raise ValueError('online source/episode order/normalization differs from the reference cache')
        if a.mmap_images:
            dataset.set_mmap_allow_build(False)
        embedder = CLIPTextEmbedder(cfg.clip, device='cpu')
    else:
        dataset, collate_fn = cache, torch.utils.data.default_collate
    if a.mode in ('features', 'live'):
        loader, sampler = make_loader(dataset, config=cfg, distributed=False, train=True, collate_fn=collate_fn)
        stream = batches_forever(loader, sampler)

    def convert(raw):
        if online:
            return policy_batch_from_loader(raw, camera_keys=cfg.model.camera_keys, device=device, dtype=dtype,
                                            embedder=embedder, train=True, mask_state_ratio=0,
                                            state_dim=cfg.model.state_dim, action_dim=cfg.model.action_dim,
                                            action_steps=cfg.model.chunk_length)
        return tensor_tree(move_batch_to_device(raw, device, non_blocking=True),
                           lambda t: t.to(dtype=dtype) if t.is_floating_point() else t)

    if stream is None:
        keys = next(iter(CursorBatchSampler(dataset, a.batch, seed=cfg.seed)))
        indices = [key[2] for key in keys]
        samples = (dataset.__getitems__(indices) if isinstance(dataset, FeatureDataset)
                   else [dataset[i] for i in indices])
        resident = convert(collate_fn(samples))
        del samples
        if a.mode.startswith('synthetic'):
            resident = tensor_tree(resident, lambda t: torch.randn_like(t) if t.is_floating_point() else t)
            resident['action_mask'] = torch.ones_like(resident['actions'], dtype=torch.bool)
            resident['camera_mask'] = torch.ones(a.batch, len(cfg.model.camera_keys), device=device, dtype=torch.bool)
            if a.mode == 'synthetic-images':
                resident.pop('vision_features')
                resident['images'] = {cam: torch.randn(a.batch, 3, 224, 224, device=device, dtype=dtype)
                                      for cam in cfg.model.camera_keys}
                if cfg.model.history_size != 1:
                    raise ValueError('synthetic image comparison currently requires history_size=1')

    timer = StageTimer()
    vision_events = []
    original_encode = model.encode_vision_features

    def encode(images):
        if not timer.enabled:
            return original_encode(images)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_encode(images)
        end.record()
        vision_events.append((start, end))
        return result
    model.encode_vision_features = encode
    observed = []

    def step():
        with timer.stage('loader_wait'):
            raw = next(stream) if stream is not None else None
        with timer.stage('prepare_and_transfer'):
            batch = convert(raw) if stream is not None else resident
            batch = mask_state(batch, cfg.flow.mask_state_ratio)
        with timer.stage('forward'):
            loss = model(batch, max_action_prefix=cfg.flow.max_action_prefix,
                         prefix_conditioning_prob=cfg.flow.prefix_conditioning_prob,
                         prefix_noise_scale=cfg.flow.prefix_noise_scale,
                         split_prefix_conditioning=a.split_prefix)
        with timer.stage('backward_and_zero_grad'):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        with timer.stage('gradient_clipping'):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
        with timer.stage('optimizer'):
            optimizer.step()
        with timer.stage('scheduler'):
            scheduler.step()
        torch.cuda.synchronize()
        observed.extend((loss.detach(), norm.detach()))

    # Reset training RNG after setup so input preparation does not change the seed.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    warmup_start = time.perf_counter()
    for _ in range(a.warmup):
        step()
    warmup_seconds = time.perf_counter() - warmup_start
    torch.cuda.reset_peak_memory_stats()
    timer.track_loader = True
    step_seconds = []
    start = time.perf_counter()
    for i in range(a.steps):
        step_start = time.perf_counter()
        step()
        step_seconds.append(time.perf_counter() - step_start)
        if (i + 1) % 10 == 0:
            print(f'MEASURED {i + 1}/{a.steps}', flush=True)
    seconds = time.perf_counter() - start
    timer.track_loader = False
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    assert torch.isfinite(torch.stack(observed)).all(), 'nonfinite loss or gradient norm'
    timer.enabled = True
    for _ in range(a.profile_steps):
        step()
    assert torch.isfinite(torch.stack(observed)).all(), 'nonfinite diagnostic loss or gradient norm'
    timer.enabled = False
    if a.kernels:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
        output.with_suffix('.kernels.txt').write_text(
            prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35))
    trainable, total = count_trainable(model)
    result = dict(mode=a.mode, mmap_images=a.mmap_images, contract=spec, contract_sha256=digest,
                  fused_adamw=a.fused_adamw, attention_backend=a.attention, split_prefix=a.split_prefix,
                  compile_conditioning=a.compile_conditioning, warmup_seconds=warmup_seconds,
                  cache_manifest_sha256=cache.fingerprint, rows=len(dataset), workers=a.workers,
                  prefetch_factor=cfg.data.prefetch_factor,
                  warmup=a.warmup, measured_updates=a.steps, seconds=seconds,
                  samples_per_second=a.steps*a.batch/seconds, peak_allocated_mib=peak_mib,
                  trainable_parameters=trainable, total_parameters=total,
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  measured_loader_wait_ms={k: float(fn(timer.loader_seconds))*1000
                                           for k, fn in [('mean', np.mean), ('p95', lambda x: np.percentile(x, 95)),
                                                         ('max', np.max)]},
                  measured_step_ms={k: float(fn(step_seconds))*1000
                                    for k, fn in [('mean', np.mean), ('p95', lambda x: np.percentile(x, 95)),
                                                  ('max', np.max)]},
                  profile_updates=a.profile_steps,
                  diagnostic_stage_ms={k: v*1000/a.profile_steps for k, v in timer.seconds.items()},
                  vision_gpu_ms_nested_in_forward=sum(s.elapsed_time(e) for s, e in vision_events)/a.profile_steps,
                  profiling_is_separate_synchronized_pass=True,
                  note='resident inputs are reused; synthetic masks are all valid; not a convergence comparison')
    atomic_json(output, result)
    print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
