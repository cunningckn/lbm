#!/usr/bin/env python3
"""Per-rank train throughput under Megatron-FSDP.

Unique global samples/s = per-rank samp/s × world_size (weak scaling).
Rank 0 emits a parseable ``RESULT`` line (used by scale_fsdp / profile_fsdp).
``--profile`` prints CUDA-event fwd / bwd / optimizer ms — the main FSDP
optimization signal.

    torchrun --standalone --nproc_per_node=8 benchmarks/throughput_fsdp.py --batch-size 32
    torchrun --standalone --nproc_per_node=8 benchmarks/throughput_fsdp.py --profile
    torchrun --standalone --nproc_per_node=1 benchmarks/throughput_fsdp.py --no-fsdp --batch-size 32
    torchrun --standalone --nproc_per_node=8 benchmarks/throughput_fsdp.py --ddp --batch-size 32
"""

from __future__ import annotations

import argparse
import os

import torch
from _common import parse_ints, peak_mem_mb, profile_train_step, try_oom

from lbm import DiTConfig, DiTPolicy, ParallelConfig, make_fake_batch, validate_model_config
from lbm.config import (
    OptimConfig,
    add_encoder_arguments,
    add_fsdp_wrap_arguments,
    apply_encoder_args,
    apply_fsdp_wrap_args,
    encoder_train_summary,
    fsdp_wrap_summary,
)
from lbm.distributed import init_distributed, wrap_policy_fsdp
from lbm.models.attention import configure_torch_sdp, set_use_flash_attn
from lbm.optim import build_adamw
from lbm.utils.bench import time_calls
from lbm.utils.gpu_util import GpuUtilSampler, format_util, gather_util


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=None, help="single per-rank batch")
    p.add_argument("--batch-sizes", default="", help="comma-separated per-rank batches")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fsdp", action="store_true", help="single-process dense baseline")
    p.add_argument("--ddp", action="store_true", help="replicate with torch DDP (no FSDP shard)")
    p.add_argument("--shard-dino", action="store_true")
    p.add_argument("--no-double-buffer", action="store_true")
    p.add_argument("--no-overlap", action="store_true")
    p.add_argument("--grad-comm-dtype", default="bf16", choices=("bf16", "fp32", "auto"))
    add_fsdp_wrap_arguments(p)
    p.add_argument(
        "--profile",
        action="store_true",
        help="CUDA-event split of fwd/bwd/optimizer (keep this on when tuning FSDP)",
    )
    p.add_argument(
        "--util",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="sample GPU/SM util during the timed window",
    )
    p.add_argument("--bf16", default=True, action=argparse.BooleanOptionalAction)
    add_encoder_arguments(p)
    return p.parse_args()


def all_ranks_oom(local_oom: bool, device: torch.device, used_dist: bool) -> bool:
    if not used_dist:
        return local_oom
    flag = torch.tensor([1 if local_oom else 0], device=device, dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch)
    loss.backward()
    optimizer.step()
    return loss


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.no_fsdp and args.ddp:
        raise SystemExit("cannot combine --ddp and --no-fsdp")

    if args.no_fsdp:
        if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise SystemExit("--no-fsdp is a 1-GPU baseline; do not launch with nproc>1")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise SystemExit("CUDA required")
        rank, world, local_rank = 0, 1, 0
        used_dist = False
        used_fsdp = False
        mode = "dense"
    else:
        local_rank, rank, world = init_distributed()
        device = torch.device("cuda", local_rank)
        used_dist = True
        used_fsdp = not args.ddp
        mode = "ddp" if args.ddp else "fsdp"

    config = DiTConfig()
    apply_encoder_args(config, args)
    parallel = ParallelConfig(
        shard_dino=args.shard_dino,
        fsdp_double_buffer=not args.no_double_buffer,
        overlap_param_gather=not args.no_overlap,
        overlap_grad_reduce=not args.no_overlap,
        grad_comm_dtype=args.grad_comm_dtype,
    )
    apply_fsdp_wrap_args(parallel, args)
    errors = validate_model_config(config)
    if errors:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))

    use_bf16 = bool(args.bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    configure_torch_sdp(mode="auto")
    set_use_flash_attn(None)

    model = DiTPolicy(config).to(device=device, dtype=dtype)
    if used_fsdp:
        optimizer = build_adamw(model, OptimConfig())
        model, optimizer = wrap_policy_fsdp(model, optimizer, parallel, device=device)
    elif args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        optimizer = build_adamw(model, OptimConfig())
    else:
        optimizer = build_adamw(model, OptimConfig())

    model.train()
    if args.batch_sizes.strip():
        batch_sizes = parse_ints(args.batch_sizes)
    elif args.batch_size is not None:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = [32]

    if rank == 0:
        print(
            f"world={world} mode={mode} fsdp={int(used_fsdp)} batch_sizes={batch_sizes} "
            f"bf16={use_bf16} {fsdp_wrap_summary(parallel)} "
            f"{encoder_train_summary(config)}"
        )
        print(
            f"{'bs':>4}  {'per_rank':>10}  {'unique':>10}  {'Hz':>8}  "
            f"{'ms/call':>10}  {'peak_MiB':>8}  {'gpu%':>6}  {'sm%':>6}  {'wave%':>6}"
        )

    oom_row = (
        f"{'OOM':>10}  {'OOM':>10}  {'OOM':>8}  {'OOM':>10}  {'OOM':>8}  "
        f"{'OOM':>6}  {'OOM':>6}  {'OOM':>6}"
    )

    for bs in batch_sizes:
        batch = make_fake_batch(config, bs, device=device, dtype=dtype)
        torch.cuda.reset_peak_memory_stats(device)
        oom = all_ranks_oom(
            try_oom(lambda batch=batch: train_step(model, optimizer, batch)),
            device,
            used_dist,
        )
        if oom:
            if rank == 0:
                print(f"{bs:>4}  {oom_row}")
                print(
                    "RESULT "
                    f"world={world} mode={mode} fsdp={int(used_fsdp)} bs={bs} "
                    f"per_rank=nan unique=nan peak_mib=nan ms=nan oom=1"
                )
            del batch
            torch.cuda.empty_cache()
            break

        loss = train_step(model, optimizer, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss {loss} at bs={bs}")

        sampler = GpuUtilSampler(local_rank) if args.util else None
        if sampler is not None:
            sampler.start()
        stats = time_calls(
            lambda batch=batch: train_step(model, optimizer, batch),
            warmup=args.warmup,
            iters=args.iters,
            batch_size=bs,
            device=device,
        )
        local_util = sampler.stop() if sampler is not None else None
        n_fast, n_slow = sampler.n_samples() if sampler is not None else (0, 0)
        util = gather_util(local_util, enabled=used_dist)
        unique = stats.samples_per_sec * world
        mem = peak_mem_mb(device) or 0.0
        split = (
            profile_train_step(model, optimizer, batch, device, warmup=1, iters=3)
            if args.profile
            else None
        )

        if used_dist:
            torch.distributed.barrier()

        def cell(name: str) -> str:
            if util is None:
                return "—"
            val = getattr(util, name)
            return f"{val:.0f}" if val is not None else "—"

        if rank == 0:
            print(
                f"{bs:>4}  {stats.samples_per_sec:10.2f}  {unique:10.2f}  "
                f"{stats.hz:8.3f}  {stats.ms_per_call:10.2f}  {mem:8.0f}  "
                f"{cell('gpu'):>6}  {cell('sm'):>6}  {cell('wave'):>6}"
            )
            extra = ""
            if util is not None:
                extra = " " + format_util(util) + f" nfast={n_fast} nsm={n_slow}"
            if split is not None:
                print(
                    f"     profile_ms fwd={split['fwd']:.2f} bwd={split['bwd']:.2f} "
                    f"step={split['step']:.2f}"
                )
            print(
                "RESULT "
                f"world={world} mode={mode} fsdp={int(used_fsdp)} bs={bs} "
                f"per_rank={stats.samples_per_sec:.4f} unique={unique:.4f} "
                f"peak_mib={mem:.1f} ms={stats.ms_per_call:.3f} "
                f"unit={parallel.fsdp_unit} group={parallel.fsdp_group_size} "
                f"overlap={int(parallel.overlap_param_gather)} "
                f"double_buffer={int(parallel.fsdp_double_buffer)} "
                f"independent_ag={int(parallel.independent_ag_group)} "
                f"fine_gather={int(parallel.fine_grained_param_gather)} "
                f"n_units={getattr(model, '_lbm_fsdp_units', 0)} "
                f"prefetch={getattr(model, '_lbm_ag_prefetch', 'auto')} "
                f"maxpool={int(getattr(model, '_lbm_maxpool_db', 0))}"
                + extra
                + (
                    f" fwd_ms={split['fwd']:.3f} bwd_ms={split['bwd']:.3f} step_ms={split['step']:.3f}"
                    if split is not None
                    else ""
                )
            )
        del batch
        torch.cuda.empty_cache()

    if used_dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
