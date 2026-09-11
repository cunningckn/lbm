#!/usr/bin/env python3
"""Single-GPU train throughput on synthetic data.

    PYTHONPATH=src python benchmarks/throughput.py
    PYTHONPATH=src python benchmarks/throughput.py --batch-sizes 1,2,4
"""

from __future__ import annotations

import argparse

import torch
from _common import parse_ints, peak_mem_mb, reset_peak, resolve_device, try_oom

from lbm import DiTConfig, DiTPolicy, make_fake_batch, validate_model_config
from lbm.config import (
    OptimConfig,
    add_encoder_arguments,
    apply_encoder_args,
    encoder_train_summary,
)
from lbm.models.attention import configure_torch_sdp, set_use_flash_attn
from lbm.optim import build_adamw, count_trainable
from lbm.utils.bench import time_calls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-sizes", default="1")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--bf16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="bf16 on CUDA (default). --no-bf16 for fp32",
    )
    p.add_argument("--attn", choices=("auto", "flash", "math"), default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    add_encoder_arguments(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    batch_sizes = parse_ints(args.batch_sizes)
    config = DiTConfig()
    apply_encoder_args(config, args)
    errors = validate_model_config(config)
    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))

    use_bf16 = bool(args.bf16) and device.type == "cuda"
    if args.bf16 and device.type != "cuda":
        print("bf16 requested but CUDA is unavailable; using fp32")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    configure_torch_sdp(mode="auto")
    set_use_flash_attn(None if args.attn == "auto" else args.attn == "flash")

    model = DiTPolicy(config).to(device=device, dtype=dtype)
    if args.compile:
        model = torch.compile(model)
    n_train, n_params = count_trainable(model)
    print(
        f"device={device}  params={n_params / 1e6:.1f}M trainable={n_train / 1e6:.1f}M  "
        f"bf16={use_bf16}  compile={args.compile}  attn={args.attn}  "
        f"chunk={config.chunk_length}  warmup={args.warmup}  iters={args.iters}  "
        f"{encoder_train_summary(config)}"
    )
    print(f"{'bs':>4}  {'samp/s':>10}  {'Hz':>8}  {'ms/step':>10}  {'peak_MiB':>8}")

    optimizer = build_adamw(model, OptimConfig())
    model.train()

    def train_step(batch):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)
        loss.backward()
        optimizer.step()
        return loss

    for bs in batch_sizes:
        batch = make_fake_batch(config, bs, image_size=args.image_size, device=device, dtype=dtype)
        reset_peak(device)
        if try_oom(lambda: train_step(batch)):
            print(f"{bs:>4}  {'OOM':>10}  {'OOM':>8}  {'OOM':>10}  {'OOM':>8}")
            break
        stats = time_calls(
            lambda: train_step(batch),
            warmup=args.warmup,
            iters=args.iters,
            batch_size=bs,
            device=device,
        )
        mem = peak_mem_mb(device)
        mem_s = f"{mem:.0f}" if mem is not None else "—"
        print(
            f"{bs:>4}  {stats.samples_per_sec:10.2f}  {stats.hz:8.2f}  "
            f"{stats.ms_per_call:10.2f}  {mem_s:>8}"
        )


if __name__ == "__main__":
    main()
