"""LIBERO 4-in-1 LeRobot (Franka). Parquet keys: image, wrist_image, state, actions."""

from lbm.action_space import delta_eef
from lbm.dataloader.custom.datasets._lerobot_bind import bind_lerobot

NAME, SPEC, scan, read_vectors, read_frames = bind_lerobot(
    "libero",
    "franka",
    ("image", "wrist_image"),
    8,
    7,
    10.0,
    kind="lerobot",
    state_columns=("state",),
    action_columns=("actions",),
    action_space=delta_eef(format="xyz_rotvec"),
)
