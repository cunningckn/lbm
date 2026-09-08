"""Shared LeRobot dump wiring: one spec + the common scan/read trio."""

from lbm.dataloader.custom.common.lerobot import read_lerobot_frames, read_lerobot_vectors, scan_lerobot
from lbm.dataloader.custom.spec import make_spec


def bind_lerobot(*args, **kwargs):
    spec = make_spec(*args, **kwargs)
    return spec.name, spec, scan_lerobot, read_lerobot_vectors, read_lerobot_frames
