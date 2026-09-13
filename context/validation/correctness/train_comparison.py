"""Paired-seed production training with fixed held-out noise and resource telemetry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from benchmarks.numerical_parity import VARIANTS, parameter_digest

from lbm import train_loop
from lbm.config import TrainConfig
from lbm.training_features import FeatureDataset


def append_json(path, record):
    with path.open('a') as out:
        out.write(json.dumps(record) + '\n')


def linux_rss(pid, proc_root=Path('/proc')):
    """Read RSS for the process and descendants, tolerating worker exit races."""
    pending, seen, resident = [str(pid)], set(), {}
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        root = proc_root / current
        try:
            status = (root / 'status').read_text()
            children_files = list((root / 'task').glob('*/children'))
        except OSError:
            continue
        resident[current] = next((int(line.split()[1]) * 1024 for line in status.splitlines()
                                  if line.startswith('VmRSS:')), 0)
        for children in children_files:
            try:
                pending.extend(children.read_text().split())
            except OSError:
                continue
    return dict(rss=resident.get(str(pid), 0),
                worker_rss=sum(value for key, value in resident.items() if key != str(pid)))


def memory_sample():
    sample = dict(time=time.time(), gpu_allocated=torch.cuda.memory_allocated(),
                  gpu_reserved=torch.cuda.memory_reserved(), gpu_peak=torch.cuda.max_memory_allocated())
    try:
        import psutil
    except ImportError:
        psutil = None
    if Path('/proc/self/status').exists():
        sample.update(linux_rss(os.getpid()))
    elif psutil is not None:
        process = psutil.Process()
        sample['rss'] = process.memory_info().rss
        sample['worker_rss'] = 0
        for child in process.children(recursive=True):
            try:
                sample['worker_rss'] += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # DataLoader workers can exit between enumeration and sampling.
                continue
    root = Path('/sys/fs/cgroup')
    if (root / 'memory.current').exists():
        sample['container_memory'] = int((root / 'memory.current').read_text())
        sample['memory_events'] = {k: int(v) for k, v in
                                   (line.split() for line in (root / 'memory.events').read_text().splitlines())}
        stats = {k: int(v) for k, v in (line.split() for line in (root / 'memory.stat').read_text().splitlines())}
        limit = (root / 'memory.max').read_text().strip()
        sample['container_unreclaimable_estimate'] = sum(stats.get(k, 0) for k in ('anon', 'shmem', 'kernel'))
        if limit != 'max' and sample['container_unreclaimable_estimate'] > .9 * int(limit):
            raise MemoryError('container anon/SHM/kernel usage exceeds 90% of limit')
    if Path('/dev/shm').exists():
        status = os.statvfs('/dev/shm')
        sample['shm_used'] = (status.f_blocks-status.f_bfree)*status.f_frsize
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--val-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--val-every', type=int, default=250)
    parser.add_argument('--val-batches', type=int, default=16)
    parser.add_argument('--checkpoint-every', type=int, default=1500)
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('real full-model comparison requires CUDA')
    output = Path(args.output)
    if not args.resume and output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    split, compiled, fused = VARIANTS[args.variant]
    cfg = TrainConfig(seed=args.seed, feature_cache=args.train_cache, output_dir=str(output),
                      batch_size=args.batch, num_workers=2, train_steps=args.steps, val_every=args.val_every,
                      val_batches=args.val_batches, ckpt_every=args.checkpoint_every, log_every=50,
                      compile_conditioning=compiled, fused_adamw=fused, dump_batch=False, resume=args.resume)
    cfg.data.val_dataset = args.val_cache
    offset = 0
    if args.resume:
        payload = torch.load(args.resume, mmap=True, map_location='cpu', weights_only=False)
        offset = payload['step']
        del payload
    observed = dict(step=offset)
    original_policy, original_eval = train_loop.DiTPolicy, train_loop.evaluate_actions
    original_signature, original_log = train_loop._resume_signature, train_loop.log_train_metrics
    original_clip = torch.nn.utils.clip_grad_norm_
    train_cache, val_cache = FeatureDataset(args.train_cache), FeatureDataset(args.val_cache)
    contract = dict(variant=args.variant, seed=args.seed, train_manifest=train_cache.fingerprint,
                    val_manifest=val_cache.fingerprint, batch=args.batch, validation_seed=2026)
    del train_cache, val_cache
    contract_path = output / 'comparison.json'
    if args.resume and json.loads(contract_path.read_text()) != contract:
        raise ValueError('resume comparison contract differs')
    contract_path.write_text(json.dumps(contract, indent=2))

    class ObservedPolicy(original_policy):
        def forward(self, *a, **kw):
            kw['split_prefix_conditioning'] = split
            if observed['step'] == 0:
                (output / 'initial-weights.sha256').write_text(parameter_digest(self))
            rng = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
            result = super().forward(*a, **kw)
            if result.ndim == 0:
                if not torch.isfinite(result):
                    raise ValueError('nonfinite training loss')
                observed['step'] += 1
                append_json(output / 'losses.jsonl', dict(step=observed['step'], loss=result.item(), rng_sha256=rng))
            return result

    def signature(*a, **kw):
        return dict(original_signature(*a, **kw), comparison_variant=args.variant, validation_seed=2026)

    def evaluate(model, module, loaders, to_policy, **kw):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(2026)
            sources = loaders if isinstance(loaders, dict) else {'all': loaders}
            total = torch.zeros(2, device='cuda', dtype=torch.float64)
            row = dict(step=observed['step'], sources={})
            for name, loader in sources.items():
                stats = original_eval(model, module, loader, to_policy, **kw)
                if not torch.isfinite(stats).all() or stats[1] <= 0:
                    raise ValueError('nonfinite or empty held-out evaluation')
                total += stats
                row['sources'][name] = dict(error_sum=stats[0].item(), count=stats[1].item(),
                                            reconstruction=(stats[0]/stats[1]).item())
            row['reconstruction'] = (total[0]/total[1]).item()
            append_json(output / 'validation.jsonl', row)
            print('HELDOUT', json.dumps(row), flush=True)
            return total

    def log(**kw):
        original_log(**kw)
        append_json(output / 'resources.jsonl', dict(step=kw['step'], **memory_sample()))

    def clip(*a, **kw):
        kw['error_if_nonfinite'] = True
        return original_clip(*a, **kw)

    train_loop.DiTPolicy, train_loop.evaluate_actions = ObservedPolicy, evaluate
    train_loop._resume_signature, train_loop.log_train_metrics = signature, log
    torch.nn.utils.clip_grad_norm_ = clip
    try:
        append_json(output / 'resources.jsonl', dict(step=offset, **memory_sample()))
        train_loop.main(cfg)
        if observed['step'] != args.steps:
            raise ValueError('observed update count differs from requested target')
        append_json(output / 'resources.jsonl', dict(step=args.steps, **memory_sample()))
        (output / f'completed-{args.steps}.json').write_text(json.dumps(dict(contract, steps=args.steps)))
    finally:
        train_loop.DiTPolicy, train_loop.evaluate_actions = original_policy, original_eval
        train_loop._resume_signature, train_loop.log_train_metrics = original_signature, original_log
        torch.nn.utils.clip_grad_norm_ = original_clip


if __name__ == '__main__':
    main()
