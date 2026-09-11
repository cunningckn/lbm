from dataclasses import fields

from lbm.config import TRAIN_DEFAULTS, TrainConfig
from lbm.train_cli import build_train_config, parse_args


def test_cli_and_config_share_training_defaults():
    args = parse_args(["--fake-data"])
    cfg = build_train_config(args)
    assert cfg.batch_size == TRAIN_DEFAULTS.batch_size
    assert cfg.num_workers == TRAIN_DEFAULTS.num_workers
    assert cfg.seed == TRAIN_DEFAULTS.seed
    assert cfg.log_every == TRAIN_DEFAULTS.log_every
    assert cfg.val_every == TRAIN_DEFAULTS.val_every
    assert cfg.val_batches == TRAIN_DEFAULTS.val_batches
    assert cfg.ckpt_every == TRAIN_DEFAULTS.train_steps_fake + 1
    assert cfg.optim.learning_rate == TRAIN_DEFAULTS.learning_rate
    assert cfg.data.video_backend == TRAIN_DEFAULTS.video_backend
    assert cfg.data.action_mode == TRAIN_DEFAULTS.action_mode


def test_train_defaults_are_immutable():
    assert {field.name for field in fields(TRAIN_DEFAULTS)} >= {"batch_size", "learning_rate", "ckpt_every"}
    try:
        TRAIN_DEFAULTS.batch_size = 99
    except AttributeError:
        pass
    else:
        raise AssertionError("TRAIN_DEFAULTS must be immutable")
    assert TrainConfig().batch_size == TRAIN_DEFAULTS.batch_size
