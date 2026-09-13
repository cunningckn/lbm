#!/usr/bin/env python3
"""Full-policy numerical comparisons; private reference tensors never belong in Git."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from benchmarks.matched_throughput import matched_config, tensor_tree

from lbm.feature_shards import atomic_json
from lbm.models.dit import DiTPolicy, load_pretrained
from lbm.optim import build_adamw
from lbm.training_features import FeatureDataset

VARIANTS = {
    'baseline': (False, False, False),
    'repeat': (False, False, False),
    'native': (True, False, False),
    'compiled': (False, True, False),
    'fused': (False, False, True),
    'combined': (True, True, True),
}


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_stats(actual, reference, *, actual_before=None, reference_before=None, chunk=1_048_576):
    """Bound scratch allocation; form update differences before any reductions."""
    if actual.shape != reference.shape:
        raise ValueError('comparison tensor shapes differ')
    if (actual_before is None) != (reference_before is None):
        raise ValueError('both pre-update tensors are required')
    if actual_before is not None and (actual_before.shape != actual.shape or reference_before.shape != reference.shape):
        raise ValueError('pre-update tensor shapes differ')
    x, y = actual.detach().reshape(-1), reference.detach().reshape(-1)
    xb = actual_before.reshape(-1) if actual_before is not None else None
    yb = reference_before.reshape(-1) if reference_before is not None else None
    sums = dict(error_sq=0., reference_sq=0., actual_sq=0., dot=0., max_abs=0., count=x.numel())
    for start in range(0, x.numel(), chunk):
        a = x[start:start+chunk].to(dtype=torch.float64)
        b = y[start:start+chunk].to(device=a.device, dtype=torch.float64)
        if xb is not None:
            a = a - xb[start:start+chunk].to(device=a.device, dtype=torch.float64)
            b = b - yb[start:start+chunk].to(device=a.device, dtype=torch.float64)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('nonfinite comparison tensor')
        difference = a - b
        sums['error_sq'] += difference.square().sum().item()
        sums['reference_sq'] += b.square().sum().item()
        sums['actual_sq'] += a.square().sum().item()
        sums['dot'] += (a*b).sum().item()
        sums['max_abs'] = max(sums['max_abs'], difference.abs().max().item())
    return sums


def summarize(sums):
    result = dict(sums)
    result['relative_l2'] = math.sqrt(sums['error_sq']/sums['reference_sq']) if sums['reference_sq'] else None
    denominator = math.sqrt(sums['reference_sq'] * sums['actual_sq'])
    result['cosine'] = sums['dot']/denominator if denominator else None
    return result


def aggregate(stats):
    total = {key: sum(row[key] for row in stats) for key in ('error_sq', 'reference_sq', 'actual_sq', 'dot', 'count')}
    total['max_abs'] = max((row['max_abs'] for row in stats), default=0.)
    return summarize(total)


def compare_snapshot(model, before, gradients, reference):
    result = {key: {} for key in ('gradients', 'updates', 'parameters')}
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(reference['after']) or set(gradients) != set(reference['gradients']):
        raise ValueError('parameter or gradient membership differs')
    for name, parameter in parameters.items():
        # Reduce bounded chunks in FP64 on the model device. Copying every
        # parameter back to CPU makes this diagnostic dominate the experiment.
        current = parameter.detach()
        result['parameters'][name] = tensor_stats(current, reference['after'][name])
        result['updates'][name] = tensor_stats(current, reference['after'][name],
                                               actual_before=before[name], reference_before=reference['before'][name])
        if name in gradients:
            result['gradients'][name] = tensor_stats(gradients[name].to(current.device),
                                                    reference['gradients'][name])
    return {key: dict(total=aggregate(list(rows.values())),
                      tensors={name: summarize(stats) for name, stats in rows.items()})
            for key, rows in result.items()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--checkpoint', required=True, help='trusted trained policy checkpoint')
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference', required=True, help='private snapshot directory, outside repository')
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--dtype', choices=('fp32', 'bf16'), required=True)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args(argv)
    if min(a.batch, a.steps) <= 0:
        p.error('batch and steps must be positive')
    return a


def main(argv=None):
    a = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('full-model parity requires CUDA')
    destination, reference_root = Path(a.output), Path(a.reference)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference_root.mkdir(parents=True, exist_ok=True)
    cache = FeatureDataset(a.cache)
    cfg = matched_config(cache.metadata, batch=a.batch, workers=0, data_root=a.data_root)
    cache.apply_config(cfg)
    dtype = torch.float32 if a.dtype == 'fp32' else torch.bfloat16
    # FP32 reference uses full precision, rather than silently enabling TF32.
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(a.seed)
    model = DiTPolicy(cfg.model)
    payload = load_pretrained(model, a.checkpoint)
    del payload
    model.to(device='cuda', dtype=dtype).train()
    split, compiled, fused = VARIANTS[a.variant]
    model.configure_conditioning(compile=compiled)
    optimizer = build_adamw(model, cfg.optim, fused=True if fused else None)
    cfg.optim.lr_warmup_steps = 0  # Fresh moments at the trained model's full learning rate.
    identity = dict(initial_weights_sha256=parameter_digest(model), cache=cache.fingerprint,
                    dtype=a.dtype, seed=a.seed, batch=a.batch, steps=a.steps,
                    checkpoint=str(Path(a.checkpoint).resolve()), model=cache.metadata['model'],
                    learning_rate=cfg.optim.learning_rate, max_action_prefix=cfg.flow.max_action_prefix)
    identity_path = reference_root / 'identity.json'
    if a.variant == 'baseline':
        if identity_path.exists():
            raise FileExistsError(identity_path)
        atomic_json(identity_path, identity)
    elif json.loads(identity_path.read_text()) != identity:
        raise ValueError('reference experiment contract differs')
    order = torch.randperm(len(cache), generator=torch.Generator().manual_seed(a.seed)).tolist()
    report = dict(variant=a.variant, contract=identity, parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  metric_reduction='model-device FP64, bounded chunks', steps=[])
    for step in range(a.steps):
        indices = [order[(step*a.batch+i) % len(order)] for i in range(a.batch)]
        raw = torch.utils.data.default_collate(cache.__getitems__(indices))
        batch = tensor_tree(raw, lambda t: t.to(device='cuda', dtype=dtype if t.is_floating_point() else t.dtype))
        generator = torch.Generator().manual_seed(a.seed + step + 1)
        masked = torch.rand(a.batch, generator=generator) < cfg.flow.mask_state_ratio
        batch['state_is_masked'] = masked.cuda()
        batch['state'] = batch['state'].masked_fill(masked.cuda()[:, None], 0)
        noise = torch.randn(batch['actions'].shape, generator=generator).to(device='cuda', dtype=dtype)
        times = torch.rand(a.batch, 1, 1, generator=generator).to(device='cuda', dtype=dtype)
        before = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
        torch.manual_seed(a.seed + 10_000 + step)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, noise=noise, t=times, split_prefix_conditioning=split,
                     max_action_prefix=cfg.flow.max_action_prefix,
                     prefix_conditioning_prob=cfg.flow.prefix_conditioning_prob,
                     prefix_noise_scale=cfg.flow.prefix_noise_scale)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        row = dict(step=step+1, loss=loss.item(), grad_norm=norm.item())
        snapshot_path = reference_root / f'{step+1}.pt'
        if a.variant == 'baseline':
            snapshot = dict(before=before, gradients=gradients,
                            after={name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
                            loss=loss.item())
            temporary = snapshot_path.with_suffix('.tmp')
            torch.save(snapshot, temporary)
            temporary.replace(snapshot_path)
            del snapshot
        else:
            reference = torch.load(snapshot_path, mmap=True, weights_only=True)
            row['loss_relative_error'] = abs(row['loss']-reference['loss']) / max(abs(reference['loss']), 1e-12)
            row.update(compare_snapshot(model, before, gradients, reference))
            del reference
        report['steps'].append(row)
        atomic_json(destination, report)
        summary = {key: value['total'] for key, value in row.items() if isinstance(value, dict)}
        print('PARITY', a.variant, a.dtype, step+1, row['loss'], json.dumps(summary), flush=True)
        del gradients, before, batch
    report['complete'] = True
    atomic_json(destination, report)


if __name__ == '__main__':
    main()
