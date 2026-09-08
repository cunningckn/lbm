"""Galaxea LeRobot v2 with split state/action columns."""

from lbm.action_space import ABS, DELTA, GRIPPER, JOINT, QUANTILE, ActionSlice
from lbm.dataloader.custom.datasets._lerobot_bind import bind_lerobot

# Packed action is L/R arm then grippers. State is ee_pose, grippers, then 7-D arms.
_GALAXEA_SPACE = (
    ActionSlice("left_arm", 0, 6, DELTA, JOINT, QUANTILE, state_start=16, state_end=22),
    ActionSlice("right_arm", 6, 12, DELTA, JOINT, QUANTILE, state_start=23, state_end=29),
    ActionSlice("left_gripper", 12, 13, ABS, GRIPPER, QUANTILE, state_start=14, state_end=15),
    ActionSlice("right_gripper", 13, 14, ABS, GRIPPER, QUANTILE, state_start=15, state_end=16),
)

NAME, SPEC, scan, read_vectors, read_frames = bind_lerobot(
    "galaxea",
    "galaxea",
    ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"),
    32,
    14,
    15.0,
    11,
    state_columns=(
        "observation.state.left_ee_pose",
        "observation.state.right_ee_pose",
        "observation.state.left_gripper",
        "observation.state.right_gripper",
        "observation.state.left_arm",
        "observation.state.right_arm",
        "observation.state.torso",
    ),
    action_columns=(
        "action.left_arm",
        "action.right_arm",
        "action.left_gripper",
        "action.right_gripper",
    ),
    kind="lerobot",
    action_space=_GALAXEA_SPACE,
)
