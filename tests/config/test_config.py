import argparse

from lbm.config import (
    DiTConfig,
    add_encoder_arguments,
    apply_encoder_args,
    encoder_train_summary,
    validate_model_config,
)


def test_default_config():
    cfg = DiTConfig()
    assert validate_model_config(cfg) == []
    assert cfg.hidden_size == 1536
    assert cfg.depth == 32
    assert cfg.num_heads == 24
    assert cfg.action_length == 5.0
    assert cfg.action_freq == 10.0
    assert cfg.chunk_length == 50
    assert cfg.history_length == 0.0
    assert cfg.history_size == 1
    assert cfg.train_vision_encoder is False
    assert cfg.train_language_encoder is False


def test_rejects_incompatible_heads():
    errors = validate_model_config(DiTConfig(num_heads=5))
    assert errors


def test_rejects_empty_cameras():
    errors = validate_model_config(DiTConfig(camera_keys=()))
    assert errors


def test_rejects_non_positive_dims():
    errors = validate_model_config(DiTConfig(depth=0))
    assert errors


def test_rejects_unknown_encoders():
    assert any("vision_encoder" in e for e in validate_model_config(DiTConfig(vision_encoder="resnet")))
    assert any("language_encoder" in e for e in validate_model_config(DiTConfig(language_encoder="bert")))
    assert any("t5_d_model" in e for e in validate_model_config(DiTConfig(t5_d_model=512, t5_num_heads=7)))


def test_apply_encoder_args_t5_and_clip():
    parser = argparse.ArgumentParser()
    add_encoder_arguments(parser)

    t5_args = parser.parse_args(["--language-encoder", "t5", "--no-train-language-encoder"])
    t5_cfg = apply_encoder_args(DiTConfig(language_max_length=200), t5_args)
    assert t5_cfg.language_encoder == "t5"
    assert t5_cfg.task_embed_dim == t5_cfg.t5_d_model
    assert t5_cfg.language_max_length == 128
    assert t5_cfg.train_language_encoder is False

    clip_args = parser.parse_args(["--vision-encoder", "siglip", "--language-encoder", "clip"])
    clip_cfg = apply_encoder_args(DiTConfig(), clip_args)
    assert clip_cfg.vision_encoder == "siglip"
    assert clip_cfg.language_encoder == "clip"
    assert clip_cfg.task_embed_dim == 512
    assert clip_cfg.language_max_length == 77
    assert clip_cfg.train_vision_encoder is False
    assert clip_cfg.train_language_encoder is False


def test_encoder_train_summary():
    cfg = DiTConfig(vision_encoder="siglip", language_encoder="t5", train_vision_encoder=False)
    summary = encoder_train_summary(cfg)
    assert "vision=siglip" in summary
    assert "language=t5" in summary
    assert "train_vision_encoder=0" in summary
    assert "train_language_encoder=" in summary
