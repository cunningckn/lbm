"""Utilities: preprocessing, synthetic data, CUDA-graph inference, benches."""

from lbm.utils.batch_dump import describe_loader_batch, save_loader_batch, video_fps_from_config
from lbm.utils.bench import CallStats, capture_sample_actions_graph, time_calls
from lbm.utils.fake_data import (
    FakeActionDataset,
    collate_samples,
    describe_batch,
    make_fake_batch,
    make_fake_sample,
    move_batch_to_device,
)
from lbm.utils.fast_inference import FastInferenceGraph, FastRTCInferenceGraph
from lbm.utils.preprocess import (
    compute_norm_stats,
    dump_norm_stats_path,
    load_dump_norm_stats,
    load_norm_stats,
    norm_stats_filename,
    normalize,
    parse_norm_stats,
    resize_pad_normalize,
    save_norm_stats,
    unnormalize,
)

__all__ = [
    "CallStats",
    "FakeActionDataset",
    "FastInferenceGraph",
    "FastRTCInferenceGraph",
    "capture_sample_actions_graph",
    "collate_samples",
    "compute_norm_stats",
    "describe_batch",
    "describe_loader_batch",
    "dump_norm_stats_path",
    "load_dump_norm_stats",
    "load_norm_stats",
    "norm_stats_filename",
    "make_fake_batch",
    "make_fake_sample",
    "move_batch_to_device",
    "normalize",
    "parse_norm_stats",
    "resize_pad_normalize",
    "save_loader_batch",
    "save_norm_stats",
    "time_calls",
    "unnormalize",
    "video_fps_from_config",
]
