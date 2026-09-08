import argparse

import torch

from lbm import ParallelConfig, validate_parallel_config
from lbm.config import (
    DiTConfig,
    add_fsdp_wrap_arguments,
    apply_fsdp_wrap_args,
    fsdp_wrap_summary,
)
from lbm.distributed import group_dit_blocks, prepare_fsdp_units, resolve_fsdp_unit_modules
from lbm.distributed.units import (
    auto_ag_prefetch_elements,
    inventory_module_sizes,
    needs_maxpool_double_buffer,
)
from lbm.models.dit import (
    DiTAttention,
    DiTBlock,
    DiTBlockGroup,
    DiTCrossAttention,
    DiTMlp,
    DiTPolicy,
    VisionTokenPool,
)
from tests.helpers import tiny_dit_config


def _tiny() -> DiTConfig:
    return tiny_dit_config(depth=4)


def test_group_dit_blocks_packs_evenly():
    model = DiTPolicy(_tiny())
    n = group_dit_blocks(model, 2)
    assert n == 2
    assert len(model.blocks) == 2
    assert all(isinstance(g, DiTBlockGroup) for g in model.blocks)
    assert all(len(g.blocks) == 2 for g in model.blocks)
    x = torch.randn(2, 8, 64)
    c = torch.randn(2, 64)
    vis = torch.randn(2, 6, 64)
    out = model.blocks[0](x, c, vis)
    assert out.shape == x.shape


def test_group_size_one_is_noop():
    model = DiTPolicy(_tiny())
    assert group_dit_blocks(model, 1) == 4
    assert all(isinstance(b, DiTBlock) for b in model.blocks)


def test_uneven_last_group():
    model = DiTPolicy(_tiny())
    n = group_dit_blocks(model, 3)
    assert n == 2
    assert len(model.blocks[0].blocks) == 3
    assert len(model.blocks[1].blocks) == 1


def test_resolve_units_block_vs_group_vs_attn():
    assert resolve_fsdp_unit_modules(ParallelConfig()) == [DiTBlock, VisionTokenPool]
    assert resolve_fsdp_unit_modules(ParallelConfig(fsdp_group_size=4)) == [
        DiTBlockGroup,
        VisionTokenPool,
    ]
    units = resolve_fsdp_unit_modules(ParallelConfig(fsdp_unit="attn_mlp"))
    assert DiTAttention in units and DiTCrossAttention in units and DiTMlp in units
    assert DiTBlock not in units
    assert VisionTokenPool in units


def test_prepare_fsdp_units_groups_before_wrap():
    model = DiTPolicy(_tiny())
    prepare_fsdp_units(model, ParallelConfig(fsdp_group_size=2))
    assert isinstance(model.blocks[0], DiTBlockGroup)


def test_maxpool_for_asymmetric_units():
    assert needs_maxpool_double_buffer(ParallelConfig(), depth=32)
    assert needs_maxpool_double_buffer(ParallelConfig(fsdp_unit="attn_mlp"), depth=32)
    assert needs_maxpool_double_buffer(ParallelConfig(shard_dino=True), depth=32)
    assert needs_maxpool_double_buffer(ParallelConfig(fsdp_group_size=3), depth=32)
    assert needs_maxpool_double_buffer(ParallelConfig(fsdp_group_size=4), depth=32)


def test_auto_prefetch_only_when_no_double_buffer():
    model = DiTPolicy(_tiny())
    units = [DiTBlock]
    assert auto_ag_prefetch_elements(model, ParallelConfig(), units) is None
    elems = auto_ag_prefetch_elements(model, ParallelConfig(fsdp_double_buffer=False), units)
    assert elems is not None and elems > 0
    forced = auto_ag_prefetch_elements(model, ParallelConfig(ag_prefetch_elements=12345), units)
    assert forced == 12345


def test_inventory_lists_dit_submodules():
    rows = {r.name: r for r in inventory_module_sizes(DiTPolicy(_tiny()))}
    assert rows["DiTBlock"].count == 4
    assert "  DiTAttention" in rows
    assert "  adaLN_modulation" in rows
    assert "  FSDP dense bucket" in rows
    assert "  FSDP expert bucket" not in rows
    assert "VisionTokenPool" in rows
    assert rows["DiTPolicy"].numel > rows["DiTBlock"].numel


def test_parallel_config_rejects_attn_mlp_with_groups():
    errors = validate_parallel_config(ParallelConfig(fsdp_unit="attn_mlp", fsdp_group_size=2))
    assert errors


def test_fsdp_wrap_cli_roundtrip():
    parser = argparse.ArgumentParser()
    add_fsdp_wrap_arguments(parser)
    args = parser.parse_args(
        ["--fsdp-group-size", "8", "--independent-ag", "--ag-prefetch-elements", "1000"]
    )
    parallel = apply_fsdp_wrap_args(ParallelConfig(), args)
    assert parallel.fsdp_group_size == 8
    assert parallel.independent_ag_group is True
    assert parallel.ag_prefetch_elements == 1000
    assert "group=8" in fsdp_wrap_summary(parallel)
