#!/usr/bin/env python3
"""Measure end-to-end synthetic training throughput.

The benchmark intentionally creates a fresh batch for each measured step so
``end_to_end_samples_per_sec`` includes input preparation. ``compute_*`` uses a
reused batch and isolates model/optimizer work. Output can be consumed as JSON:

    PYTHONPATH=src python benchmarks/train_throughput.py --device cuda --json
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

try:
    from ._common import parse_ints, peak_mem_mb, reset_peak, resolve_device
except ImportError:  # direct ``python benchmarks/train_throughput.py`` execution
    from _common import parse_ints, peak_mem_mb, reset_peak, resolve_device

from lbm import DiTConfig, DiTPolicy, make_fake_batch, validate_model_config
from lbm.config import OptimConfig, add_encoder_arguments, apply_encoder_args
from lbm.models.attention import configure_torch_sdp, set_use_flash_attn
from lbm.optim import build_adamw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per batch size")
    parser.add_argument("--attn", choices=("auto", "flash", "math"), default="auto")
    parser.add_argument("--bf16", default=True, action=argparse.BooleanOptionalAction)
    add_encoder_arguments(parser)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be >= 0 and --iters must be positive")
    if any(batch_size <= 0 for batch_size in parse_ints(args.batch_sizes)):
        parser.error("--batch-sizes must contain positive integers")
    return args


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch)
    loss.backward()
    optimizer.step()
    return loss


def measure_batch(config: DiTConfig, *, batch_size: int, device: torch.device, dtype: torch.dtype,
                  warmup: int, iters: int, compile: bool) -> dict[str, Any]:
    model = DiTPolicy(config).to(device=device, dtype=dtype)
    if compile:
        model = torch.compile(model)
    optimizer = build_adamw(model, OptimConfig())
    model.train()
    for _ in range(warmup):
        _step(model, optimizer, make_fake_batch(config, batch_size, device=device, dtype=dtype))
    _sync(device)
    reset_peak(device)
    reused = make_fake_batch(config, batch_size, device=device, dtype=dtype)
    for _ in range(warmup):
        _step(model, optimizer, reused)
    _sync(device)
    compute_start = time.perf_counter()
    for _ in range(iters):
        _step(model, optimizer, reused)
    _sync(device)
    compute_s = time.perf_counter() - compute_start
    data_s = 0.0
    _sync(device)
    end_to_end_start = time.perf_counter()
    for _ in range(iters):
        data_start = time.perf_counter()
        batch = make_fake_batch(config, batch_size, device=device, dtype=dtype)
        _sync(device)
        data_s += time.perf_counter() - data_start
        _step(model, optimizer, batch)
    _sync(device)
    end_to_end_s = time.perf_counter() - end_to_end_start
    compute_steps_s = iters / max(compute_s, 1e-12)
    end_to_end_steps_s = iters / max(end_to_end_s, 1e-12)
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "compute_steps_per_sec": compute_steps_s,
        "compute_samples_per_sec": compute_steps_s * batch_size,
        "end_to_end_steps_per_sec": end_to_end_steps_s,
        "end_to_end_samples_per_sec": end_to_end_steps_s * batch_size,
        "data_wait_ms_per_step": 1000.0 * data_s / iters,
        "data_wait_fraction": min(1.0, data_s / max(end_to_end_s, 1e-12)),
        "peak_memory_mib": peak_mem_mb(device),
    }
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    config = DiTConfig()
    apply_encoder_args(config, args)
    errors = validate_model_config(config)
    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))
    use_bf16 = bool(args.bf16) and device.type == "cuda"
    configure_torch_sdp(mode="auto")
    set_use_flash_attn(None if args.attn == "auto" else args.attn == "flash")
    for batch_size in parse_ints(args.batch_sizes):
        result = {
            "device": str(device),
            "dtype": "bfloat16" if use_bf16 else "float32",
            "compile": bool(args.compile),
            "attn": args.attn,
            **measure_batch(config, batch_size=batch_size, device=device,
                            dtype=torch.bfloat16 if use_bf16 else torch.float32,
                            warmup=args.warmup, iters=args.iters, compile=args.compile),
        }
        print(json.dumps(result, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
