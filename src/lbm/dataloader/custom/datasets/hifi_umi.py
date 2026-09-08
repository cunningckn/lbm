"""HiFi UMI LeRobot v3 packed."""

from lbm.action_space import dual_eef
from lbm.dataloader.custom.datasets._lerobot_bind import bind_lerobot

NAME, SPEC, scan, read_vectors, read_frames = bind_lerobot(
    "hifi_umi",
    "hifi_umi",
    ("head_main", "left_hand_up", "right_hand_up"),
    20,
    20,
    25.0,
    16,
    kind="lerobot_v3",
    action_space=dual_eef(eef=9, sides=("right", "left")),
)
