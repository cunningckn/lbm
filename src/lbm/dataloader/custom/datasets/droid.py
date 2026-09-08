"""DROID LeRobot v2.1 (Franka). Packed 8-D joint+gripper; two exteriors + wrist."""

from lbm.action_space import unimanual_joint
from lbm.dataloader.custom.datasets._lerobot_bind import bind_lerobot

NAME, SPEC, scan, read_vectors, read_frames = bind_lerobot(
    "droid",
    "oxe_droid",
    ("exterior_1_left", "exterior_2_left", "wrist_left"),
    8,
    8,
    15.0,
    kind="lerobot",
    action_space=unimanual_joint(),
)
