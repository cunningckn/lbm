"""Wrap DiTPolicy with Megatron-FSDP. Lazy-imports megatron_fsdp."""

from __future__ import annotations

from typing import Sequence, Type

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from lbm.config import ParallelConfig, validate_parallel_config
from lbm.distributed.units import (
    apply_ag_prefetch,
    auto_ag_prefetch_elements,
    count_fsdp_units,
    make_independent_ag_groups,
    needs_maxpool_double_buffer,
    prepare_fsdp_units,
    resolve_fsdp_unit_modules,
)
from lbm.models.dino import DinoBlock
from lbm.models.dit import DiTBlock, VisionTokenPool
from lbm.models.siglip import SiglipBlock

_DTYPE = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def default_fsdp_unit_modules(*, shard_dino: bool = False) -> list[Type[nn.Module]]:
    """Outermost compute tiles Megatron-FSDP may unshard then reshard.

    DiTBlock is depth-wise symmetric. ``VisionTokenPool`` is a different size
    and must be a unit so Megatron-FSDP all-gathers queries on ``forward()``
    (a ``ParameterDict`` indexed in Python stays sharded otherwise). DINO
    blocks are another size — only include them with MaxPool double-buffering.
    Embeddings, ``pos_embed``, and frozen ``img_proj`` stay unsharded.
    """
    units: list[Type[nn.Module]] = [DiTBlock, VisionTokenPool]
    if shard_dino:
        units.extend((DinoBlock, SiglipBlock))
    return units


def _policy_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    if name not in _DTYPE:
        raise ValueError(f"unknown dtype name {name!r}")
    return _DTYPE[name]


def _import_megatron_fsdp():
    try:
        import importlib

        from megatron_fsdp import MixedPrecisionPolicy, fully_shard
        from megatron_fsdp.distributed_data_parallel_config import (
            DistributedDataParallelConfig as FsdpDDPConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "megatron-fsdp is required for FSDP training. Install with: "
            "uv sync --extra dist"
        ) from exc
    # DAS megatron-core 0.15.4 DDPConfig is older than megatron-fsdp 0.6.
    # fully_shard prefers the mcore class when both are installed, then passes
    # 0.6-only fields. Import the submodules (not the package-level function
    # aliases) and point them at the standalone config.
    for name in (
        "megatron_fsdp.fully_shard",
        "megatron_fsdp.megatron_fsdp",
        "megatron_fsdp.param_and_grad_buffer",
    ):
        importlib.import_module(name).DistributedDataParallelConfig = FsdpDDPConfig
    return MixedPrecisionPolicy, fully_shard


def wrap_fsdp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    parallel: ParallelConfig,
    device_mesh: DeviceMesh,
    *,
    fsdp_unit_modules: Sequence[Type[nn.Module]] | None = None,
    device: torch.device | None = None,
    fsdp_group_ag=None,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Fully-shard ``model`` and rewrite ``optimizer`` onto sharded parameters."""
    errors = validate_parallel_config(parallel)
    if errors:
        raise ValueError("Invalid parallel config:\n  - " + "\n  - ".join(errors))

    MixedPrecisionPolicy, fully_shard = _import_megatron_fsdp()
    units = (
        list(fsdp_unit_modules)
        if fsdp_unit_modules is not None
        else resolve_fsdp_unit_modules(parallel)
    )
    mixed = MixedPrecisionPolicy(
        main_params_dtype=_policy_dtype(parallel.main_params_dtype),
        main_grads_dtype=_policy_dtype(parallel.main_grads_dtype),
        grad_comm_dtype=_policy_dtype(parallel.grad_comm_dtype),
    )
    depth = getattr(getattr(model, "config", None), "depth", None)
    if parallel.maxpool_double_buffer is None:
        use_maxpool = parallel.fsdp_double_buffer and needs_maxpool_double_buffer(
            parallel, depth=depth
        )
    else:
        use_maxpool = bool(parallel.maxpool_double_buffer) and parallel.fsdp_double_buffer
    prefetch_elems = auto_ag_prefetch_elements(model, parallel, units)
    n_units = count_fsdp_units(model, units)
    model, optimizer = fully_shard(
        module=model,
        optimizer=optimizer,
        device_mesh=device_mesh,
        dp_shard_dim=parallel.dp_shard_dim,
        tp_dim=parallel.tp_dim,
        expt_device_mesh=device_mesh,
        fsdp_group_ag=fsdp_group_ag,
        expt_fsdp_group_ag=fsdp_group_ag,
        fsdp_unit_modules=units,
        zero_dp_strategy=parallel.zero_dp_strategy,
        device=device,
        mixed_precision_policy=mixed,
        overlap_grad_reduce=parallel.overlap_grad_reduce,
        overlap_param_gather=parallel.overlap_param_gather,
        fsdp_double_buffer=parallel.fsdp_double_buffer,
        maxpool_double_buffer=use_maxpool,
        average_in_collective=parallel.average_in_collective,
        enable_fine_grained_param_gather=parallel.fine_grained_param_gather,
    )
    applied = apply_ag_prefetch(model, prefetch_elems)
    model._lbm_fsdp_units = n_units
    model._lbm_ag_prefetch = applied
    model._lbm_maxpool_db = use_maxpool
    return model, optimizer


def wrap_policy_fsdp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    parallel: ParallelConfig,
    *,
    device: torch.device | None = None,
    fsdp_unit_modules: Sequence[Type[nn.Module]] | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Group DiT blocks, then fully-shard on the dense DP mesh."""
    from lbm.distributed.mesh import build_fsdp_meshes

    prepare_fsdp_units(model, parallel)
    meshes = build_fsdp_meshes(parallel)
    fsdp_group_ag = make_independent_ag_groups(meshes.dense, parallel)
    return wrap_fsdp(
        model,
        optimizer,
        parallel,
        meshes.dense,
        fsdp_unit_modules=fsdp_unit_modules,
        device=device,
        fsdp_group_ag=fsdp_group_ag,
    )
