"""Megatron-FSDP wrap helpers (no multi-GPU required except the spawn test)."""

from __future__ import annotations

import os
import socket

import pytest
import torch

from lbm import ParallelConfig, validate_parallel_config
from lbm.config import DiTConfig, OptimConfig
from lbm.distributed import default_fsdp_unit_modules, prepare_fsdp_env
from lbm.models.dino import DinoBlock
from lbm.models.dit import DiTBlock, DiTPolicy, VisionTokenPool
from tests.helpers import tiny_dit_config


def test_default_fsdp_units_include_vision_pool():
    units = default_fsdp_unit_modules()
    assert units == [DiTBlock, VisionTokenPool]
    assert DinoBlock in default_fsdp_unit_modules(shard_dino=True)
    assert VisionTokenPool in default_fsdp_unit_modules(shard_dino=True)


def test_prepare_fsdp_env_unsets_max_connections(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    prepare_fsdp_env()
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ


def test_parallel_config_rejects_bad_zero_strategy():
    errors = validate_parallel_config(ParallelConfig(zero_dp_strategy="zero3"))
    assert errors


def test_parallel_config_defaults_ok():
    assert validate_parallel_config(ParallelConfig()) == []


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tiny_config() -> DiTConfig:
    return tiny_dit_config()


def _fsdp_worker(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["PYTHONUNBUFFERED"] = "1"

    from lbm.distributed import build_dp_mesh, init_distributed, wrap_fsdp
    from lbm.optim import build_adamw
    from lbm.utils.fake_data import make_fake_batch

    local_rank, _, _ = init_distributed()
    device = torch.device("cuda", local_rank)
    config = _tiny_config()
    dtype = torch.bfloat16
    model = DiTPolicy(config).to(device=device, dtype=dtype)
    optimizer = build_adamw(model, OptimConfig(learning_rate=1e-4))
    mesh = build_dp_mesh()
    model, optimizer = wrap_fsdp(model, optimizer, ParallelConfig(), mesh, device=device)
    batch = make_fake_batch(config, batch_size=2, device=device, dtype=dtype)
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="need 2 GPUs")
def test_fsdp_two_gpu_train_step():
    pytest.importorskip("megatron_fsdp")
    world = 2
    port = _free_port()
    torch.multiprocessing.spawn(_fsdp_worker, args=(world, port), nprocs=world, join=True)
