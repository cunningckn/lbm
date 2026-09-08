"""RobotWin LeRobot (ALOHA / Agilex wrists). Nested dumps under ``datasets/robotwin``."""

from lbm.action_space import bimanual_joint
from lbm.dataloader.custom.datasets._lerobot_bind import bind_lerobot
from lbm.dataloader.custom.spec import WRISTS

NAME, SPEC, scan, read_vectors, read_frames = bind_lerobot(
    "robotwin", "aloha", WRISTS, 14, 14, 30.0, kind="lerobot", action_space=bimanual_joint()
)
