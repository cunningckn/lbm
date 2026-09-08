"""Shared test helpers."""

from __future__ import annotations

from lbm.config import DiTConfig


def tiny_dit_config(**kwargs) -> DiTConfig:
    """Small DiT for unit tests (override any field via kwargs)."""
    cfg = dict(
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        action_length=8.0,
        action_freq=1.0,
        vit_embed_dim=64,
        vit_depth=1,
        vit_num_heads=4,
        vision_pool_num_queries=2,
        vision_pool_num_heads=4,
        language_encoder="none",
        t5_vocab_size=128,
        t5_d_model=64,
        t5_d_ff=128,
        t5_num_layers=1,
        t5_num_heads=4,
        language_max_length=8,
        task_embed_dim=64,
    )
    cfg.update(kwargs)
    return DiTConfig(**cfg)
