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

NAME, SPEC, _scan, read_vectors, read_frames = bind_lerobot(
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
    scan_revision=2,
)


def _instruction_text(text):
    """Galaxea's Chinese@English annotation supplies an English CLIP instruction."""
    before, separator, after = text.partition('@')
    if separator and any('\u4e00' <= char <= '\u9fff' for char in before) and after.strip():
        return after.strip()
    return text


def scan(root, spec, *, max_episodes=None):
    records = _scan(root, spec, max_episodes=max_episodes)
    for record in records:
        record.lang = _instruction_text(record.lang)
    return records


def read_subtasks(record):
    """Decode fine task_index changes from physical rows, without reading video payloads."""
    import numpy as np
    import pandas as pd

    from lbm.dataloader.custom.common.lerobot import _read_episode_parquet, lerobot_of
    from lbm.dataloader.custom.common.lerobot_rows import load_tasks
    from lbm.dataloader.custom.instructions import from_frame_tasks

    dump = lerobot_of(record)
    if dump is None:
        raise ValueError('Galaxea subtask mode requires a LeRobot record')
    columns = ['task_index', 'frame_index']
    frame = (_read_episode_parquet(dump.parquet, dump.episode_index, columns=columns) if dump.is_v3
             else pd.read_parquet(dump.parquet, columns=columns))
    if len(frame) != record.n_frames or not np.array_equal(frame['frame_index'], np.arange(record.n_frames)):
        raise ValueError('Galaxea subtask rows do not match physical episode frame coordinates')
    tasks = {key: _instruction_text(text) for key, text in load_tasks(dump.repo/'meta').items()}
    return from_frame_tasks(frame['task_index'].to_numpy(), tasks)
