from dataclasses import asdict

import pytest
import torch
from benchmarks.matched_throughput import contract, mask_state, matched_config, parse_args

from lbm.config import TrainConfig


def test_cache_recipe_controls_all_matched_modes():
    source = TrainConfig()
    source.model.action_freq = 30
    source.model.state_dim = 20
    source.model.action_dim = 22
    cfg = matched_config({'source_config': asdict(source)}, batch=128, workers=2, data_root='/data')
    assert not cfg.data.use_mmap and not cfg.data.use_mmap_frames
    assert cfg.model.chunk_length == 150
    assert cfg.flow.max_action_prefix == 4
    baseline, digest = contract(cfg)
    cfg.num_workers = 8
    assert contract(cfg) == (baseline, digest)
    cfg.flow.max_action_prefix = 0
    assert contract(cfg)[1] != digest
    cfg.flow.max_action_prefix = 4
    cfg.model.action_freq = 10
    assert contract(cfg)[1] != digest


def test_masking_does_not_mutate_resident_inputs():
    source = dict(state=torch.ones(4, 20), actions=torch.ones(4, 150, 22))
    masked = mask_state(source, 1.0)
    assert not masked['state'].any()
    assert source['state'].all()
    assert 'state_is_masked' not in source
    assert masked['actions'] is source['actions']
    assert mask_state(source, 0)['state'] is source['state']


@pytest.mark.parametrize('flag,value', [('--batch', '0'), ('--steps', '0'), ('--workers', '-1'),
                                       ('--warmup', '-1'), ('--profile-steps', '0'), ('--prefetch-factor', '0')])
def test_invalid_benchmark_bounds(flag, value):
    with pytest.raises(SystemExit):
        parse_args(['--mode', 'features', '--cache', '/cache', '--data-root', '/data',
                    '--output', '/report.json', flag, value])


def test_mmap_mode_requires_online_images():
    base = ['--cache', '/cache', '--data-root', '/data', '--output', '/report.json', '--mmap-images']
    with pytest.raises(SystemExit):
        parse_args(['--mode', 'features', *base])
    assert parse_args(['--mode', 'live', *base]).mmap_images
