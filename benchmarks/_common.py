"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FSDP_BENCH = Path(__file__).resolve().parent / "throughput_fsdp.py"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def peak_mem_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def try_oom(fn) -> bool:
    """Run ``fn()``. Return True on CUDA OOM, re-raise other errors."""
    try:
        fn()
        return False
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except RuntimeError as exc:
        text = str(exc).lower()
        if "out of memory" in text or "out_of_resources" in text:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        raise


def cuda_ms(fn, *, device: torch.device, warmup: int, iters: int) -> float:
    """Mean CUDA-event milliseconds over ``iters`` calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / iters


def profile_train_step(
    model, optimizer, batch, device: torch.device, *, warmup: int = 2, iters: int = 3
) -> dict[str, float]:
    """CUDA-event split of one train step: fwd / bwd / optimizer."""
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    ev = torch.cuda.Event
    pairs = {name: (ev(True), ev(True)) for name in ("fwd", "bwd", "step")}
    acc = {name: 0.0 for name in pairs}
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        pairs["fwd"][0].record()
        loss = model(batch)
        pairs["fwd"][1].record()
        pairs["bwd"][0].record()
        loss.backward()
        pairs["bwd"][1].record()
        pairs["step"][0].record()
        optimizer.step()
        pairs["step"][1].record()
        torch.cuda.synchronize(device)
        for name, (a, b) in pairs.items():
            acc[name] += a.elapsed_time(b)
    return {name: acc[name] / iters for name in acc}


def parse_result(stdout: str) -> dict[str, str]:
    """Parse the last ``RESULT key=val ...`` line from a FSDP bench run."""
    found: dict[str, str] | None = None
    for line in stdout.splitlines():
        if not line.startswith("RESULT "):
            continue
        kv: dict[str, str] = {}
        for tok in line.split()[1:]:
            key, _, val = tok.partition("=")
            kv[key] = val
        found = kv
    if found is None:
        raise RuntimeError("no RESULT line in:\n" + stdout[-2000:])
    return found


def torchrun_fsdp(
    extra: list[str],
    *,
    nproc: int,
    profile: bool = False,
    util: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(FSDP_BENCH),
        *extra,
    ]
    if profile:
        cmd.append("--profile")
    if not util:
        cmd.append("--no-util")
    env = os.environ.copy()
    env.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
    src = str(SRC)
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
    env.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    env.setdefault("NCCL_DEBUG", "WARN")
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
