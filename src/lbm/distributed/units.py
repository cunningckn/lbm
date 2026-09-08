"""FSDP unit selection, DiT block grouping, and module-size inventory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Type

import torch
from torch import nn

from lbm.config import ParallelConfig
from lbm.models.dino import DinoBlock
from lbm.models.dit import (
    DiTAttention,
    DiTBlock,
    DiTBlockGroup,
    DiTCrossAttention,
    DiTMlp,
    VisionTokenPool,
)
from lbm.models.siglip import SiglipBlock


def group_dit_blocks(model: nn.Module, group_size: int) -> int:
    """Replace ``model.blocks`` with ``DiTBlockGroup``s. Returns the group count.

    ``group_size == 1`` is a no-op. The last group may be smaller when
    ``len(blocks)`` is not divisible by ``group_size``.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError(f"{type(model).__name__} has no .blocks to group")
    if group_size == 1:
        return len(blocks)
    if len(blocks) > 0 and isinstance(blocks[0], DiTBlockGroup):
        return len(blocks)
    grouped: list[DiTBlockGroup] = []
    raw = list(blocks)
    for i in range(0, len(raw), group_size):
        grouped.append(DiTBlockGroup(raw[i : i + group_size]))
    model.blocks = nn.ModuleList(grouped)
    return len(grouped)


def prepare_fsdp_units(model: nn.Module, parallel: ParallelConfig) -> int:
    """Apply wrap granularity to ``model`` before ``fully_shard``. Returns group count."""
    if parallel.fsdp_unit == "attn_mlp":
        return len(getattr(model, "blocks", []))
    return group_dit_blocks(model, parallel.fsdp_group_size)


def needs_maxpool_double_buffer(
    parallel: ParallelConfig, *, depth: int | None = None
) -> bool:
    """Asymmetric FSDP units cannot share a fixed-size double buffer.

    ``VisionTokenPool`` is always a unit and is a different size from
    ``DiTBlock``, so MaxPool is required even for the default wrap. Sharded
    DINO, attn/mlp split, and uneven groups are additional cases.
    """
    del parallel, depth
    return True


def resolve_fsdp_unit_modules(
    parallel: ParallelConfig,
) -> list[Type[nn.Module]]:
    """Classes Megatron-FSDP treats as unshard / reshard tiles."""
    if parallel.fsdp_unit == "attn_mlp":
        units: list[Type[nn.Module]] = [DiTAttention, DiTCrossAttention, DiTMlp]
    elif parallel.fsdp_group_size > 1:
        units = [DiTBlockGroup]
    else:
        units = [DiTBlock]
    units.append(VisionTokenPool)
    if parallel.shard_dino:
        units.extend((DinoBlock, SiglipBlock))
    return units


def count_fsdp_units(model: nn.Module, unit_modules: Sequence[Type[nn.Module]]) -> int:
    types = tuple(unit_modules)
    return sum(1 for m in model.modules() if isinstance(m, types))


def mean_unit_numel(model: nn.Module, unit_modules: Sequence[Type[nn.Module]]) -> int:
    types = tuple(unit_modules)
    sizes = [sum(p.numel() for p in m.parameters()) for m in model.modules() if isinstance(m, types)]
    if not sizes:
        return 0
    return int(sum(sizes) / len(sizes))


def auto_ag_prefetch_elements(
    model: nn.Module,
    parallel: ParallelConfig,
    unit_modules: Sequence[Type[nn.Module]],
) -> int | None:
    """Prefetch budget Megatron-FSDP should actually use.

    The library floors ``suggested_communication_unit_size`` at 1e9 elements.
    With double-buffering that floor is irrelevant (in-flight units cap at 2).
    Without it, the floor would unshard far more than "current + next".
    """
    if parallel.ag_prefetch_elements is not None:
        return parallel.ag_prefetch_elements
    if parallel.fsdp_double_buffer:
        return None
    unit = mean_unit_numel(model, unit_modules)
    if unit <= 0:
        return None
    return unit * 2


def apply_ag_prefetch(fsdp_model: nn.Module, elements: int | None) -> int | None:
    """Override Megatron-FSDP's all-gather prefetch size after ``fully_shard``."""
    if elements is None:
        return getattr(fsdp_model, "suggested_AG_prefetch_size", None)
    if not hasattr(fsdp_model, "suggested_AG_prefetch_size"):
        raise TypeError(f"{type(fsdp_model).__name__} is not a MegatronFSDP wrap")
    fsdp_model.suggested_AG_prefetch_size = int(elements)
    fsdp_model.suggested_RS_queue_capacity = int(elements) * 2
    return int(elements)


def make_independent_ag_groups(
    dense_mesh,
    parallel: ParallelConfig,
):
    """Duplicate the DP group so param all-gather and grad reduce-scatter can overlap.

    All ranks must call this. Returns ``None`` when independent AG is off.
    """
    import torch.distributed as dist

    if not parallel.independent_ag_group:
        return None
    dp = parallel.dp_shard_dim
    dense_ranks = list(dist.get_process_group_ranks(dense_mesh[dp].get_group()))
    return dist.new_group(ranks=dense_ranks)


@dataclass(frozen=True)
class ModuleSize:
    name: str
    count: int
    numel: int

    @property
    def mparams(self) -> float:
        return self.numel / 1e6

    @property
    def ag_mib_unsharded(self) -> float:
        return self.numel * 2 / (1024**2)

    def ag_mib_sharded(self, world: int) -> float:
        if world < 1:
            return self.ag_mib_unsharded
        return self.numel * 2 / world / (1024**2)


def _named_first(model: nn.Module, cls: type) -> nn.Module | None:
    for m in model.modules():
        if type(m) is cls or isinstance(m, cls):
            return m
    return None


def inventory_module_sizes(model: nn.Module) -> list[ModuleSize]:
    """Param counts for wrap candidates. One representative of each class plus totals."""
    rows: list[ModuleSize] = []

    def add(name: str, module: nn.Module | None, count: int = 1) -> None:
        if module is None:
            return
        rows.append(
            ModuleSize(name=name, count=count, numel=sum(p.numel() for p in module.parameters()))
        )

    n_blocks = sum(1 for m in model.modules() if type(m) is DiTBlock)
    first_block = _named_first(model, DiTBlock)
    add("DiTBlock", first_block, n_blocks)
    if first_block is not None:
        add("  DiTAttention", first_block.attn)
        add("  DiTCrossAttention", first_block.cross_attn)
        add("  DiTMlp", first_block.mlp)
        add("  adaLN_modulation", first_block.adaLN_modulation)
        rows.append(
            ModuleSize(
                name="  FSDP dense bucket",
                count=1,
                numel=sum(p.numel() for p in first_block.parameters()),
            )
        )
    first_group = _named_first(model, DiTBlockGroup)
    n_groups = sum(1 for m in model.modules() if isinstance(m, DiTBlockGroup))
    add("DiTBlockGroup", first_group, n_groups)
    add("VisionTokenPool", getattr(model, "vision_pool", None))
    add(
        "DinoBlock",
        _named_first(model, DinoBlock),
        sum(1 for m in model.modules() if isinstance(m, DinoBlock)),
    )
    add(
        "SiglipBlock",
        _named_first(model, SiglipBlock),
        sum(1 for m in model.modules() if isinstance(m, SiglipBlock)),
    )
    apool = getattr(model, "apool", None)
    if apool is not None and len(apool):
        add("AttentionPool (1 cam)", next(iter(apool.values())), len(apool))
    add("img_backbone", getattr(model, "img_backbone", None))
    add("DiTPolicy", model)
    dit_blocks = getattr(model, "blocks", None)
    if dit_blocks is not None:
        add("all DiT units (blocks/groups)", dit_blocks, 1)
    return rows


def format_inventory(rows: Sequence[ModuleSize], *, world: int = 8) -> str:
    lines = [
        f"{'module':<28} {'n':>4}  {'Mparams':>8}  {'AG MiB':>8}  "
        f"{'AG/rank@'+str(world):>12}  {'all n×Mparams':>14}"
    ]
    for row in rows:
        lines.append(
            f"{row.name:<28} {row.count:4d}  {row.mparams:8.2f}  "
            f"{row.ag_mib_unsharded:8.2f}  {row.ag_mib_sharded(world):12.2f}  "
            f"{row.mparams * row.count:14.1f}"
        )
    return "\n".join(lines)


@torch.no_grad()
def _cuda_ms(fn, *, device: torch.device, warmup: int, iters: int) -> float:
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


def time_dit_submodules(
    model: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int = 2,
    iters: int = 8,
) -> dict[str, float]:
    """Isolated CUDA-event times for wrap-granularity decisions (dense, 1 GPU)."""
    from lbm.utils.fake_data import make_fake_batch

    config = model.config
    batch = make_fake_batch(config, batch_size, device=device, dtype=dtype)
    vision = model.build_vision_tokens(batch["images"])
    x_t = torch.randn(
        batch_size, config.chunk_length, config.action_dim, device=device, dtype=dtype
    )
    t = torch.rand(batch_size, device=device, dtype=dtype)
    task_h = model.encode_task(batch)
    c = model.compute_cond(batch["state"], task_h, t)
    z = model.y_embedder(x_t) + model.pos_embed.data[:, : x_t.shape[1], :]
    block = model.blocks[0]
    inner = block.blocks[0] if isinstance(block, DiTBlockGroup) else block

    def _clone(x):
        return x.detach()

    times = {
        "vision_fwd": _cuda_ms(
            lambda: model.build_vision_tokens(batch["images"]),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        "one_unit_fwd": _cuda_ms(
            lambda: block(_clone(z), _clone(c), _clone(vision)),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        "one_block_fwd": _cuda_ms(
            lambda: inner(_clone(z), _clone(c), _clone(vision)),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        "attn_fwd": _cuda_ms(
            lambda: inner.attn(_clone(z)),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        "xattn_fwd": _cuda_ms(
            lambda: inner.cross_attn(_clone(z), _clone(vision)),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        "mlp_fwd": _cuda_ms(
            lambda: inner.mlp(_clone(z)),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
        "all_dit_fwd": _cuda_ms(
            lambda: model.predict_velocity(_clone(x_t), _clone(c), _clone(vision)),
            device=device,
            warmup=warmup,
            iters=iters,
        ),
    }
    return times
