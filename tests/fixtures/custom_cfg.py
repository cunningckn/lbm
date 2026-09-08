"""Required keys for ``make_custom_dataset`` — no silent .get defaults."""

from lbm.action_space import ABS


def custom_cfg(**overrides) -> dict:
    cfg = {
        "action_length": 0.2,
        "history_length": 0.0,
        "use_mmap": False,
        "use_mmap_frames": False,
        "action_mode": ABS,
        "rescan": False,
    }
    cfg.update(overrides)
    return cfg
