"""Kai0 nested LeRobot v2 (ALOHA)."""

from lbm.action_space import bimanual_joint
from lbm.dataloader.custom.datasets._lerobot_bind import bind_lerobot

NAME, SPEC, scan, read_vectors, read_frames = bind_lerobot(
    "kai0",
    "aloha",
    ("top_head", "hand_left", "hand_right"),
    14,
    14,
    30.0,
    7,
    kind="lerobot",
    action_space=bimanual_joint(),
)
