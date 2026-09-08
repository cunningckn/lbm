"""Torch distributed init and DeviceMesh for Megatron-FSDP."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from lbm.config import ParallelConfig


def prepare_fsdp_env() -> None:
    """FSDP overlap needs independent CUDA streams (NVIDIA recommended)."""
    os.environ.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)


def init_distributed(*, backend: str | None = None) -> tuple[int, int, int]:
    """Initialize the process group and bind this rank to its GPU.

    Returns ``(local_rank, rank, world_size)``.
    """
    prepare_fsdp_env()
    if not torch.cuda.is_available():
        raise RuntimeError("Megatron-FSDP training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend or "nccl",
            device_id=torch.device("cuda", local_rank),
        )
    return local_rank, torch.distributed.get_rank(), torch.distributed.get_world_size()


def build_dp_mesh(
    *,
    world_size: int | None = None,
    mesh_dim_name: str = "dp_shard",
    tp_dim_name: str = "tp",
    device_type: str = "cuda",
) -> DeviceMesh:
    """DP × TP=1 mesh. Megatron-FSDP requires a TP axis even when TP is unused."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("build_dp_mesh requires init_distributed() first")
    size = world_size if world_size is not None else torch.distributed.get_world_size()
    if size < 1:
        raise ValueError(f"world_size must be >= 1, got {size}")
    return init_device_mesh(
        device_type,
        (size, 1),
        mesh_dim_names=(mesh_dim_name, tp_dim_name),
    )


@dataclass
class FSDPMeshes:
    """Dense ``(dp_shard, tp=1)`` FSDP mesh."""

    dense: DeviceMesh


def build_fsdp_meshes(
    parallel: ParallelConfig,
    *,
    world_size: int | None = None,
    device_type: str = "cuda",
) -> FSDPMeshes:
    """Build the dense ``(dp_shard, tp=1)`` mesh."""
    dense = build_dp_mesh(
        world_size=world_size,
        mesh_dim_name=parallel.dp_shard_dim,
        tp_dim_name=parallel.tp_dim,
        device_type=device_type,
    )
    return FSDPMeshes(dense=dense)
