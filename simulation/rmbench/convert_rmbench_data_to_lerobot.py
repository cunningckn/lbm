"""Convert RMBench raw demos directly to LeRobot format.

Usage (from ``simulation/lerobot``)::

    # All 12 tasks → lbm/datasets/rmbench/
    bash ../rmbench/convert_rmbench_data_to_lerobot.sh
    uv run python ../rmbench/convert_rmbench_data_to_lerobot.py 50

    # M(1) / M(n)
    uv run python ../rmbench/convert_rmbench_data_to_lerobot.py 50 --task-set m1
    uv run python ../rmbench/convert_rmbench_data_to_lerobot.py 50 --task-set mn

    # Single task (writes to lbm/datasets/rmbench/<task>/)
    uv run python ../rmbench/convert_rmbench_data_to_lerobot.py cover_blocks 50

Raw:     ``lbm/third_party/rmbench/data/<task>/<setting>/`` (default setting: demo_clean)
LeRobot: ``lbm/datasets/rmbench/``, ``rmbench_m1/``, or ``rmbench_mn/``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Literal

import cv2
import h5py
import numpy as np
import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

_SIMULATION_DIR = Path(__file__).resolve().parents[1]
if str(_SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_DIR))
from paths import datasets_dir, rmbench_root  # noqa: E402

DEFAULT_RMBENCH = rmbench_root()
DEFAULT_OUTPUT_DIR_ALL = datasets_dir() / "rmbench"
DEFAULT_REPO_ID_ALL = "rmbench"
DEFAULT_OUTPUT_DIR_M1 = datasets_dir() / "rmbench_m1"
DEFAULT_REPO_ID_M1 = "rmbench_m1"
DEFAULT_OUTPUT_DIR_MN = datasets_dir() / "rmbench_mn"
DEFAULT_REPO_ID_MN = "rmbench_mn"

ALL_TASKS = [
    "battery_try",
    "blocks_ranking_try",
    "classify_blocks",
    "cover_blocks",
    "observe_and_pickup",
    "place_block_mat",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "storage_blocks",
    "swap_T",
    "swap_blocks",
]

# M(1) tasks (Mem-0 / RMBench): single-stage execution, typically trained jointly.
M1_TASKS = [
    "observe_and_pickup",
    "put_back_block",
    "rearrange_blocks",
    "swap_blocks",
    "swap_T",
]

# M(n) tasks (Mem-0 / RMBench): multi-stage; typically trained per task.
MN_TASKS = [
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "press_button",
    "place_block_mat",
]

# RMBench camera name → LeRobot / Aloha camera name used by LBM training.
CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def _prepare_output(root: Path, *, overwrite: bool) -> None:
    if not root.exists():
        return
    if root.is_symlink() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite symlink {root} -> {root.resolve()}. "
            "Pass --overwrite or --output-dir <new path>."
        )
    if not overwrite:
        raise SystemExit(f"Output already exists: {root}. Pass --overwrite to replace it.")
    if root.is_symlink() or root.is_file():
        root.unlink()
    else:
        shutil.rmtree(root)


def create_empty_dataset(
    repo_id: str,
    *,
    robot_type: str = "aloha",
    mode: Literal["video", "image"] = "video",
    output_dir: Path | None = None,
    image_writer_processes: int = 10,
    image_writer_threads: int = 5,
    overwrite: bool = False,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }
    for cam in CAMERA_MAP.values():
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    root = Path(output_dir) if output_dir is not None else HF_LEROBOT_HOME / repo_id
    _prepare_output(root, overwrite=overwrite)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=root,
        robot_type=robot_type,
        features=features,
        use_videos=mode == "video",
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def _decode_resize(jpeg_bytes: bytes, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to decode jpeg frame")
    return cv2.resize(frame, size)


def load_raw_episode(ep_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load one RMBench episode → (images, state, action) with Aloha-style shift.

    At timestep t: state/images from t, action from t+1. Length is T-1.
    """
    with h5py.File(ep_path, "r") as root:
        # joint_action/vector is [left_arm(6), left_gripper, right_arm(6), right_gripper]
        qpos = np.asarray(root["/joint_action/vector"][()], dtype=np.float32)
        if qpos.shape[0] < 2:
            raise ValueError(f"episode too short ({qpos.shape[0]} steps): {ep_path}")

        imgs: dict[str, list[np.ndarray]] = {name: [] for name in CAMERA_MAP.values()}
        # Images align with state (all but last timestep).
        for t in range(qpos.shape[0] - 1):
            for raw_cam, lerobot_cam in CAMERA_MAP.items():
                bits = root[f"/observation/{raw_cam}/rgb"][t]
                imgs[lerobot_cam].append(_decode_resize(bits))

    images = {cam: np.stack(frames) for cam, frames in imgs.items()}
    state = qpos[:-1]
    action = qpos[1:]
    return images, state, action


def populate_task(
    dataset: LeRobotDataset,
    load_dir: Path,
    episode_num: int,
    *,
    desc: str,
) -> tuple[int, list[int]]:
    skipped: list[int] = []
    written = 0

    for i in tqdm.tqdm(range(episode_num), desc=desc):
        raw_hdf5 = load_dir / "data" / f"episode{i}.hdf5"
        instruction_path = load_dir / "instructions" / f"episode{i}.json"
        try:
            with instruction_path.open("r", encoding="utf-8") as f_instr:
                instructions = json.load(f_instr)["seen"]
            images, state, action = load_raw_episode(raw_hdf5)
        except (OSError, KeyError, json.JSONDecodeError, FileNotFoundError, ValueError) as e:
            logging.warning("skip corrupt/missing episode %d (%s): %s", i, raw_hdf5, e)
            skipped.append(i)
            continue

        instruction = np.random.choice(instructions)
        for t in range(state.shape[0]):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "task": instruction,
            }
            for cam, img_array in images.items():
                frame[f"observation.images.{cam}"] = img_array[t]
            dataset.add_frame(frame)
        dataset.save_episode()
        written += 1

    if skipped:
        logging.warning("skipped %d episodes for %s: %s", len(skipped), desc, skipped)
    return written, skipped


def convert_tasks(
    task_names: list[str],
    episode_num: int,
    repo_id: str,
    *,
    rmbench_root: Path,
    setting: str,
    output_dir: Path | None = None,
    mode: Literal["video", "image"] = "video",
    push_to_hub: bool = False,
    overwrite: bool = False,
) -> Path:
    dataset = create_empty_dataset(repo_id, mode=mode, output_dir=output_dir, overwrite=overwrite)
    total_written = 0

    for task_name in task_names:
        load_dir = rmbench_root / "data" / task_name / setting
        if not (load_dir / "data").is_dir():
            logging.warning("skip task (no raw data): %s", load_dir)
            continue

        print(f"read raw: {load_dir}")
        written, _ = populate_task(
            dataset,
            load_dir,
            episode_num,
            desc=f"{repo_id}/{task_name}",
        )
        if written == 0:
            logging.warning("no episodes written for task %s", task_name)
            continue
        total_written += written
        print(f"  wrote {written}/{episode_num} episodes for {task_name}")

    if total_written == 0:
        raise RuntimeError(f"No episodes written for {repo_id} from {rmbench_root}")

    print(f"Wrote {total_written} episodes total → {dataset.root} (mode={mode})")
    if push_to_hub:
        dataset.push_to_hub()
    return Path(dataset.root)


def main():
    parser = argparse.ArgumentParser(description="Convert RMBench raw demos to LeRobot.")
    parser.add_argument(
        "task_name",
        type=str,
        nargs="?",
        default=None,
        help="Single task name (e.g. cover_blocks). Default: convert a task set into one dataset.",
    )
    parser.add_argument(
        "--task-set",
        type=str,
        choices=("all", "m1", "mn"),
        default="all",
        help="Task split when task_name is omitted: all (12), m1 (5), or mn (5)",
    )
    parser.add_argument(
        "expert_data_num",
        type=int,
        nargs="?",
        default=50,
        help="Number of episodes per task (default: 50)",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="demo_clean",
        help="Raw data setting subdirectory (default: demo_clean)",
    )
    parser.add_argument(
        "--rmbench-root",
        type=Path,
        default=Path(os.environ.get("RMBENCH_ROOT", DEFAULT_RMBENCH)),
        help="Path to lbm/third_party/rmbench",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="LeRobot output dir (default: lbm/datasets/rmbench, rmbench_m1, or rmbench_mn)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot repo id (default: rmbench, rmbench_m1, rmbench_mn, or rmbench/<task>)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("video", "image"),
        default="video",
        help="Image storage mode (default: video)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push LeRobot dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dir (required if the path already exists or is a symlink)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    episode_num = args.expert_data_num
    # Allow `...py 50` as shorthand for all tasks with 50 episodes per task.
    if task_name is not None and task_name.isdigit():
        episode_num = int(task_name)
        task_name = None

    if task_name is None:
        if args.task_set == "m1":
            task_names = M1_TASKS
            repo_id = args.repo_id or DEFAULT_REPO_ID_M1
            output_dir = args.output_dir or DEFAULT_OUTPUT_DIR_M1
        elif args.task_set == "mn":
            task_names = MN_TASKS
            repo_id = args.repo_id or DEFAULT_REPO_ID_MN
            output_dir = args.output_dir or DEFAULT_OUTPUT_DIR_MN
        else:
            task_names = ALL_TASKS
            repo_id = args.repo_id or DEFAULT_REPO_ID_ALL
            output_dir = args.output_dir or DEFAULT_OUTPUT_DIR_ALL
    else:
        task_names = [task_name]
        repo_id = args.repo_id or f"rmbench/{task_name}"
        output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR_ALL / task_name)

    print(f"tasks: {', '.join(task_names)}")
    print(f"write LeRobot: {output_dir} (repo_id={repo_id}, mode={args.mode})")
    convert_tasks(
        task_names,
        episode_num,
        repo_id,
        rmbench_root=args.rmbench_root,
        setting=args.setting,
        output_dir=output_dir,
        mode=args.mode,
        push_to_hub=args.push_to_hub,
        overwrite=args.overwrite,
    )
    print("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
