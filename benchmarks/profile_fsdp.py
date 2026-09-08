#!/usr/bin/env python3
"""FSDP wrap / prefetch profile — the main distributed optimization signal.

Every timed run uses ``throughput_fsdp.py --profile`` so RESULT lines include
fwd / bwd / optimizer CUDA-event ms.

Phases:
  inventory  1-GPU param inventory + isolated submodule times
  a          wrap/prefetch variants
  b          group_size and attn_mlp
  all        inventory + a + b

    PYTHONPATH=src python benchmarks/profile_fsdp.py --phase inventory
    PYTHONPATH=src python benchmarks/profile_fsdp.py --phase all --nproc 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from _common import parse_result, torchrun_fsdp

from lbm import DiTConfig, DiTPolicy, validate_model_config
from lbm.distributed.units import format_inventory, inventory_module_sizes, time_dit_submodules
from lbm.models.attention import configure_torch_sdp, set_use_flash_attn


@dataclass
class Row:
    label: str
    world: int
    mode: str
    bs: int
    unique: float
    ms: float
    peak_mib: float
    group: int = 1
    overlap: int = 1
    double_buffer: int = 1
    independent_ag: int = 0
    maxpool: int = 0
    fwd_ms: float | None = None
    bwd_ms: float | None = None
    step_ms: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", default="all", choices=("inventory", "a", "b", "all"))
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    return p.parse_args()


def run_inventory(batch_size: int) -> None:
    config = DiTConfig()
    errors = validate_model_config(config)
    if errors:
        raise SystemExit("\n".join(errors))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    configure_torch_sdp(mode="auto")
    set_use_flash_attn(None)
    model = DiTPolicy(config)
    if device.type == "cuda":
        model = model.to(device=device, dtype=dtype)
    print("=== module inventory  (bf16 AG bytes = numel×2) ===")
    print(format_inventory(inventory_module_sizes(model), world=8))
    if device.type != "cuda":
        return
    model.train()
    print(f"\n=== isolated fwd ms  bs={batch_size}  (1-GPU) ===")
    times = time_dit_submodules(model, batch_size=batch_size, device=device, dtype=dtype)
    block_ms = times["one_block_fwd"]
    print(f"{'section':<16} {'ms':>8}  {'vs 1 block':>10}")
    for name, ms in times.items():
        print(f"{name:<16} {ms:8.2f}  {ms / block_ms:10.2f}×")
    ag_elems = sum(p.numel() for p in model.blocks[0].parameters())
    ag_mib = ag_elems * 2 / 8 / (1024**2)
    print(
        f"\nrule of thumb: one DiTBlock AG on 8 ranks ≈ {ag_mib:.2f} MiB. "
        f"If one_block_fwd ({block_ms:.2f} ms) is shorter than that AG, "
        f"overlap cannot hide communication until group_size grows."
    )


def row_from_kv(kv: dict[str, str], label: str) -> Row:
    return Row(
        label=label,
        world=int(kv["world"]),
        mode=kv.get("mode", "fsdp"),
        bs=int(kv["bs"]),
        unique=float(kv["unique"]),
        ms=float(kv["ms"]),
        peak_mib=float(kv["peak_mib"]),
        group=int(kv.get("group", "1")),
        overlap=int(kv.get("overlap", "1")),
        double_buffer=int(kv.get("double_buffer", "1")),
        independent_ag=int(kv.get("independent_ag", "0")),
        maxpool=int(kv.get("maxpool", "0")),
        fwd_ms=float(kv["fwd_ms"]) if "fwd_ms" in kv else None,
        bwd_ms=float(kv["bwd_ms"]) if "bwd_ms" in kv else None,
        step_ms=float(kv["step_ms"]) if "step_ms" in kv else None,
    )


def launch(args: argparse.Namespace, extra: list[str], *, nproc: int, label: str) -> Row | None:
    cmd_extra = [
        "--batch-size",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        *extra,
    ]
    print("\n$", f"torchrun nproc={nproc} throughput_fsdp.py --profile", *extra, flush=True)
    proc = torchrun_fsdp(cmd_extra, nproc=nproc, profile=True, util=False)
    print(proc.stdout[-3500:] if len(proc.stdout) > 3500 else proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
        print(f"  -> {label} FAILED (code {proc.returncode})", flush=True)
        return None
    row = row_from_kv(parse_result(proc.stdout + proc.stderr), label)
    print(
        f"  -> unique={row.unique:.1f}  {row.ms:.1f} ms  "
        f"fwd={row.fwd_ms} bwd={row.bwd_ms} adam={row.step_ms}  "
        f"{row.peak_mib:.0f} MiB  maxpool={row.maxpool}",
        flush=True,
    )
    return row


def print_table(rows: list[Row], baseline: Row | None, *, vs: str) -> None:
    print(f"\n=== sweep vs {vs} ===")
    print(
        f"{'label':<32} {'unique':>8} {'ms':>8} {'fwd':>8} {'bwd':>8} {'adam':>8} "
        f"{'MiB':>8} {('vs ' + vs):>8}"
    )
    for row in rows:
        rel = f"{100 * row.unique / baseline.unique:5.1f}%" if baseline is not None else "—"
        fwd = f"{row.fwd_ms:.1f}" if row.fwd_ms is not None else "—"
        bwd = f"{row.bwd_ms:.1f}" if row.bwd_ms is not None else "—"
        adam = f"{row.step_ms:.1f}" if row.step_ms is not None else "—"
        print(
            f"{row.label:<32} {row.unique:8.1f} {row.ms:8.1f} {fwd:>8} {bwd:>8} "
            f"{adam:>8} {row.peak_mib:8.0f} {rel:>8}"
        )


def winner_flags(winner: Row) -> list[str]:
    flags: list[str] = []
    if winner.overlap == 0:
        flags.append("--no-overlap")
    if winner.double_buffer == 0:
        flags.append("--no-double-buffer")
    if winner.independent_ag:
        flags.append("--independent-ag")
    return flags


def phase_a(args: argparse.Namespace) -> tuple[Row | None, list[Row]]:
    labeled = [
        ("ddp", ["--ddp"]),
        ("fsdp overlap=off", ["--no-overlap"]),
        ("fsdp overlap+db", []),
        ("fsdp overlap no-db", ["--no-double-buffer"]),
        ("fsdp overlap+db+ag", ["--independent-ag"]),
    ]
    rows: list[Row] = []
    for label, extra in labeled:
        row = launch(args, extra, nproc=args.nproc, label=label)
        if row is not None:
            rows.append(row)
    ddp = next((r for r in rows if r.mode == "ddp"), None)
    fsdp_rows = [r for r in rows if r.mode != "ddp"]
    if fsdp_rows:
        winner = max(fsdp_rows, key=lambda r: r.unique)
        print(f"\nphase A winner: {winner.label} unique={winner.unique:.1f}")
    return ddp, rows


def phase_b(args: argparse.Namespace, ddp: Row, winner: Row) -> list[Row]:
    rows: list[Row] = [ddp, winner]
    flags = winner_flags(winner)
    for group in (2, 4, 8, 16, 32):
        row = launch(
            args,
            flags + ["--fsdp-group-size", str(group)],
            nproc=args.nproc,
            label=f"fsdp group={group}",
        )
        if row is not None:
            rows.append(row)
    row = launch(args, flags + ["--fsdp-unit", "attn_mlp"], nproc=args.nproc, label="fsdp unit=attn_mlp")
    if row is not None:
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.phase in ("inventory", "all"):
        run_inventory(args.batch_size)
        if args.phase == "inventory":
            return

    baseline: Row | None = None
    rows: list[Row] = []
    winner: Row | None = None

    if args.phase in ("a", "all"):
        baseline, rows = phase_a(args)
        fsdp_rows = [r for r in rows if r.mode != "ddp"]
        winner = max(fsdp_rows, key=lambda r: r.unique) if fsdp_rows else None
        print_table(rows, baseline, vs="DDP")

    if args.phase in ("b", "all"):
        if baseline is None or winner is None:
            baseline = launch(args, ["--ddp"], nproc=args.nproc, label="ddp")
            winner = launch(args, [], nproc=args.nproc, label="fsdp overlap+db")
        if baseline is None or winner is None:
            raise SystemExit("need a DDP baseline and an FSDP winner before phase B")
        rows = phase_b(args, baseline, winner)
        print_table(rows, baseline, vs="DDP")


if __name__ == "__main__":
    main()
