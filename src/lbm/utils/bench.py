"""Timing helpers for throughput and inference-frequency benches."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CallStats:
    """Result of timing `iters` calls of a function."""

    elapsed_s: float
    iters: int
    batch_size: int

    @property
    def hz(self) -> float:
        """Calls per second (inference frequency / train step frequency)."""
        return self.iters / max(self.elapsed_s, 1e-12)

    @property
    def samples_per_sec(self) -> float:
        return self.hz * self.batch_size

    @property
    def ms_per_call(self) -> float:
        return 1000.0 / self.hz


def synchronize(device: torch.device | str | None) -> None:
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_calls(
    fn,
    *,
    warmup: int,
    iters: int,
    batch_size: int,
    device: torch.device | str,
) -> CallStats:
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    for _ in range(warmup):
        fn()
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return CallStats(time.perf_counter() - t0, iters, batch_size)


def capture_sample_actions_graph(
    model: nn.Module,
    batch: dict,
    num_steps: int,
    *,
    warmup: int = 5,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
    """CUDA-graph capture of `sample_actions` with static noise.

    Returns `(graph, output, noise)`. Replay with `graph.replay()`; `output` is updated in place.
    """
    device = batch["state"].device
    if device.type != "cuda":
        raise RuntimeError("CUDA graph capture requires a CUDA batch")
    dtype = model.y_embedder.weight.dtype
    noise = torch.randn(
        batch["state"].shape[0],
        model.chunk_length,
        model.action_dim,
        device=device,
        dtype=dtype,
    )

    def infer():
        return model.sample_actions(batch, num_steps=num_steps, noise=noise)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    output = None
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            output = infer()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = infer()
    assert output is not None
    return graph, output, noise
