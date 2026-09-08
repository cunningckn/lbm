#!/usr/bin/env python3
"""Closed-loop inference frequency: ``sample_actions`` at batch size 1.

Hz = replans / s. Default paths: eager and CUDA-graph (CUDA).

    PYTHONPATH=src python benchmarks/infer_hz.py
    PYTHONPATH=src python benchmarks/infer_hz.py --paths eager,compile,cuda-graph
"""

from __future__ import annotations

import argparse

import torch

from _common import peak_mem_mb, reset_peak, resolve_device

from lbm import DiTConfig, DiTPolicy, make_fake_batch
from lbm.utils.bench import capture_sample_actions_graph, time_calls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument(
        "--paths",
        default="auto",
        help="comma: eager, compile, cuda-graph. auto = eager,cuda-graph on CUDA else eager",
    )
    p.add_argument(
        "--bf16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="bf16 on CUDA (default). --no-bf16 for fp32",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def resolve_paths(spec: str, device: torch.device) -> list[str]:
    if spec == "auto":
        return ["eager", "cuda-graph"] if device.type == "cuda" else ["eager"]
    paths = [p.strip() for p in spec.split(",") if p.strip()]
    allowed = {"eager", "compile", "cuda-graph"}
    unknown = [p for p in paths if p not in allowed]
    if unknown:
        raise SystemExit(f"unknown --paths {unknown}; expected {sorted(allowed)}")
    return paths


def build_model(config, device: torch.device, *, bf16: bool, compile_model: bool) -> DiTPolicy:
    model = DiTPolicy(config).to(device)
    if bf16:
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    if compile_model:
        model = torch.compile(model)
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    use_bf16 = bool(args.bf16) and device.type == "cuda"
    if args.bf16 and device.type != "cuda":
        print("bf16 requested but CUDA is unavailable; using fp32")
    paths = resolve_paths(args.paths, device)
    config = DiTConfig()
    n_params = sum(p.numel() for p in DiTPolicy(config).parameters())
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    batch = make_fake_batch(config, 1, device=device, dtype=dtype)

    print(
        f"device={device}  params={n_params / 1e6:.1f}M  bs=1  "
        f"steps={args.diffusion_steps}  bf16={use_bf16}  "
        f"warmup={args.warmup}  iters={args.iters}"
    )
    print("Hz = sample_actions calls / s (closed-loop replan rate)")
    print(f"{'path':<12}  {'Hz':>8}  {'ms/infer':>10}  {'peak_MiB':>8}")

    def report(path: str, stats, mem) -> None:
        mem_s = f"{mem:.0f}" if mem is not None else "—"
        if stats is None:
            print(f"{path:<12}  {'skip':>8}  {'skip':>10}  {mem_s:>8}")
            return
        print(f"{path:<12}  {stats.hz:8.2f}  {stats.ms_per_call:10.2f}  {mem_s:>8}")

    if "eager" in paths:
        model = build_model(config, device, bf16=use_bf16, compile_model=False)
        reset_peak(device)
        stats = time_calls(
            lambda: model.sample_actions(batch, num_steps=args.diffusion_steps),
            warmup=args.warmup,
            iters=args.iters,
            batch_size=1,
            device=device,
        )
        report("eager", stats, peak_mem_mb(device))
        del model

    if "compile" in paths:
        model = build_model(config, device, bf16=use_bf16, compile_model=True)
        reset_peak(device)
        stats = time_calls(
            lambda: model.sample_actions(batch, num_steps=args.diffusion_steps),
            warmup=max(args.warmup, 3),
            iters=args.iters,
            batch_size=1,
            device=device,
        )
        report("compile", stats, peak_mem_mb(device))
        del model

    if "cuda-graph" in paths:
        if device.type != "cuda":
            report("cuda-graph", None, None)
        else:
            model = build_model(config, device, bf16=use_bf16, compile_model=False)
            reset_peak(device)
            try:
                graph, _, _ = capture_sample_actions_graph(
                    model, batch, args.diffusion_steps, warmup=max(args.warmup, 5)
                )
            except Exception as exc:
                print(f"{'cuda-graph':<12}  failed: {exc}")
            else:
                stats = time_calls(
                    graph.replay,
                    warmup=args.warmup,
                    iters=args.iters,
                    batch_size=1,
                    device=device,
                )
                report("cuda-graph", stats, peak_mem_mb(device))
                del graph
            del model


if __name__ == "__main__":
    main()
