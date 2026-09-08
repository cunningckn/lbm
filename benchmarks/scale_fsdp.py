#!/usr/bin/env python3
"""Weak-scaling sweep: 1-GPU dense vs FSDP at 1/2/4/8 GPUs.

Efficiency = unique_global / (world × 1-GPU-FSDP unique). Target ≥ 0.80.
Sub-linear runs print a short diagnosis and re-run with ``--profile``.

    PYTHONPATH=src python benchmarks/scale_fsdp.py
    PYTHONPATH=src python benchmarks/scale_fsdp.py --world-sizes 1,2,4,8 --batch-size 32
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from _common import parse_ints, parse_result, torchrun_fsdp

from lbm.config import add_encoder_arguments


@dataclass
class Row:
    world: int
    fsdp: bool
    bs: int
    per_rank: float
    unique: float
    peak_mib: float
    ms: float
    fwd_ms: float | None = None
    bwd_ms: float | None = None
    step_ms: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-sizes", default="1,2,4,8")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--min-efficiency", type=float, default=0.80)
    p.add_argument("--include-dense-baseline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shard-dino", action="store_true")
    p.add_argument("--no-double-buffer", action="store_true")
    p.add_argument("--no-overlap", action="store_true")
    p.add_argument("--grad-comm-dtype", default="bf16")
    p.add_argument("--profile-on-fail", action=argparse.BooleanOptionalAction, default=True)
    add_encoder_arguments(p)
    return p.parse_args()


def row_from_kv(kv: dict[str, str]) -> Row:
    return Row(
        world=int(kv["world"]),
        fsdp=bool(int(kv["fsdp"])),
        bs=int(kv["bs"]),
        per_rank=float(kv["per_rank"]),
        unique=float(kv["unique"]),
        peak_mib=float(kv["peak_mib"]),
        ms=float(kv["ms"]),
        fwd_ms=float(kv["fwd_ms"]) if "fwd_ms" in kv else None,
        bwd_ms=float(kv["bwd_ms"]) if "bwd_ms" in kv else None,
        step_ms=float(kv["step_ms"]) if "step_ms" in kv else None,
    )


def run_job(nproc: int, extra: list[str], profile: bool = False) -> Row:
    proc = torchrun_fsdp(extra, nproc=nproc, profile=profile, util=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"torchrun nproc={nproc} failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    return row_from_kv(parse_result(proc.stdout))


def diagnose(row: Row, ref: Row, efficiency: float) -> list[str]:
    hints = [
        f"efficiency={efficiency:.2%} vs 1-GPU FSDP unique={ref.unique:.1f} samp/s",
        f"per-rank {row.per_rank:.1f} vs 1-GPU {ref.per_rank:.1f} samp/s "
        f"(comm tax {(1 - row.per_rank / max(ref.per_rank, 1e-9)) * 100:.0f}%)",
    ]
    if row.fwd_ms is not None and row.bwd_ms is not None and row.step_ms is not None:
        compute = row.fwd_ms + row.bwd_ms
        hints.append(
            f"profile ms/call fwd={row.fwd_ms:.1f} bwd={row.bwd_ms:.1f} "
            f"step={row.step_ms:.1f} (bwd/fwd={row.bwd_ms / max(row.fwd_ms, 1e-9):.2f})"
        )
        if row.bwd_ms > 2.5 * row.fwd_ms:
            hints.append(
                "backward is much slower than forward — AG/RS likely not hidden; "
                "try larger per-rank batch, keep overlap on, or shard DINO with MaxPool"
            )
        if row.step_ms > 0.3 * compute:
            hints.append("optimizer step is a large fraction of the iteration — check ZeRO-3 install")
    else:
        hints.append("re-run with --profile-on-fail to split fwd/bwd/step")
    hints.append(
        "seq=50 is short: FSDP weight AG (~100MiB/DiTBlock) can exceed GEMM; "
        "raise --batch-size until compute-bound"
    )
    return hints


def main() -> None:
    args = parse_args()
    worlds = parse_ints(args.world_sizes)
    extra = [
        "--batch-size",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--grad-comm-dtype",
        args.grad_comm_dtype,
        "--vision-encoder",
        args.vision_encoder,
        "--language-encoder",
        args.language_encoder,
    ]
    if args.shard_dino:
        extra.append("--shard-dino")
    if args.no_double_buffer:
        extra.append("--no-double-buffer")
    if args.no_overlap:
        extra.append("--no-overlap")
    extra.append("--train-vision-encoder" if args.train_vision_encoder else "--no-train-vision-encoder")
    extra.append("--train-language-encoder" if args.train_language_encoder else "--no-train-language-encoder")

    print(f"batch-size={args.batch_size} min-efficiency={args.min_efficiency:.0%}")
    print(
        f"{'setup':<16} {'world':>5} {'bs':>4} {'per_rank':>10} {'unique':>10} "
        f"{'eff':>8} {'peak_MiB':>10}"
    )

    dense_unique_1 = None
    if args.include_dense_baseline:
        row = run_job(1, extra + ["--no-fsdp"])
        dense_unique_1 = row.unique
        print(
            f"{'dense':<16} {row.world:>5} {row.bs:>4} {row.per_rank:10.2f} "
            f"{row.unique:10.2f} {'—':>8} {row.peak_mib:10.0f}"
        )

    fsdp_rows = [run_job(n, extra) for n in worlds]
    ref = min(fsdp_rows, key=lambda r: r.world)
    unique_1 = ref.unique / ref.world
    if ref.world != 1:
        print(f"note: efficiency is relative to FSDP world={ref.world}, not 1 GPU")
    worst_eff = min(
        (r.unique / (r.world * unique_1) for r in fsdp_rows if r.world > ref.world),
        default=1.0,
    )
    for row in fsdp_rows:
        eff = row.unique / (row.world * unique_1)
        vs_dense = ""
        if dense_unique_1 is not None:
            vs_dense = f"  vs_dense={row.unique / (row.world * dense_unique_1):.1%}"
        print(
            f"{'fsdp':<16} {row.world:>5} {row.bs:>4} {row.per_rank:10.2f} "
            f"{row.unique:10.2f} {eff:8.1%} {row.peak_mib:10.0f}{vs_dense}"
        )

    failed = [
        r
        for r in fsdp_rows
        if r.world > ref.world and r.unique / (r.world * unique_1) < args.min_efficiency
    ]
    if not failed:
        print(f"PASS: weak-scaling efficiency ≥ {args.min_efficiency:.0%} (worst {worst_eff:.1%})")
        return

    print(f"FAIL: {len(failed)} config(s) below {args.min_efficiency:.0%} (worst {worst_eff:.1%})")
    target = failed[-1]
    if args.profile_on_fail and target.fwd_ms is None:
        print(f"re-running world={target.world} with CUDA-event profile…")
        target = run_job(target.world, extra, profile=True)
    for line in diagnose(target, ref, target.unique / (target.world * unique_1)):
        print("  " + line)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
