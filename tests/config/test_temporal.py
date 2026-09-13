import argparse

import numpy as np
import pytest

from lbm.config import DiTConfig, add_temporal_arguments, apply_temporal_args
from lbm.temporal import delta_indices, n_steps, native_stride


def test_n_steps_and_stride():
    assert n_steps(1.0, 50.0) == 50
    assert n_steps(5.0, 10.0) == 50
    assert n_steps(0.0, 10.0) == 1
    assert n_steps(1.8, 10.0) == 18
    assert native_stride(50.0, 50.0) == 1
    assert native_stride(50.0, 10.0) == 5
    assert native_stride(50.0, 1.0) == 50
    assert n_steps(1.0, 10.0) == 10
    assert native_stride(10.0, 10.0) == 1
    assert native_stride(10.0, 50.0) == 1


def test_libero_native_fps_deltas():
    deltas = delta_indices(1.0, 10.0, 10.0, past=False)
    assert deltas.tolist() == list(range(10))


def test_action_deltas_match_legacy_chunk():
    deltas = delta_indices(1.0, 50.0, 50.0, past=False)
    assert deltas.tolist() == list(range(50))


def test_action_deltas_downsampled():
    deltas = delta_indices(5.0, 10.0, 50.0, past=False)
    assert len(deltas) == 50
    assert deltas[0] == 0
    assert deltas[1] == 5
    assert deltas[-1] == 245


def test_history_current_only():
    deltas = delta_indices(0.0, 10.0, 50.0, past=True)
    assert deltas.tolist() == [0]


def test_history_deltas_include_present():
    # 18 frames at 1 Hz on a 50 Hz log: stride 50, oldest → now.
    deltas = delta_indices(18.0, 1.0, 50.0, past=True)
    assert len(deltas) == 18
    assert deltas[-1] == 0
    assert deltas[0] == -17 * 50
    assert np.all(np.diff(deltas) == 50)


def test_apply_temporal_args_updates_chunk_length():
    parser = argparse.ArgumentParser()
    add_temporal_arguments(parser)
    args = parser.parse_args(
        ["--action-length", "5", "--action-freq", "10", "--history-length", "1.8", "--history-freq", "10"]
    )
    cfg = apply_temporal_args(DiTConfig(), args)
    assert cfg.chunk_length == 50
    assert cfg.history_size == 18


def test_dit_config_rejects_chunk_length_kwarg():
    with pytest.raises(TypeError):
        DiTConfig(chunk_length=8)


def test_new_history_cli_and_resource_limits():
    from lbm.config import validate_model_config

    parser = argparse.ArgumentParser()
    add_temporal_arguments(parser)
    cfg = apply_temporal_args(DiTConfig(), parser.parse_args([
        '--history-time-encoding', '--state-history-length', '.3', '--state-history-freq', '10',
    ]))
    assert cfg.history_time_encoding and cfg.state_history_length == .3
    assert not validate_model_config(cfg)
    cfg.state_history_length = 10.
    assert any('64' in error for error in validate_model_config(cfg))
    cfg.state_history_length = float('nan')
    assert any('state history' in error for error in validate_model_config(cfg))
