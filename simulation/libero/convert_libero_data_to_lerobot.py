"""Convert LIBERO RLDS demos to LeRobot format.

Usage (from ``simulation/lerobot``)::

    uv run python ../libero/convert_libero_data_to_lerobot.py --data-dir /path/to/modified_libero_rlds
    bash ../libero/convert_libero_data_to_lerobot.sh /path/to/modified_libero_rlds

Raw RLDS: https://huggingface.co/datasets/openvla/modified_libero_rlds
LeRobot:  ``lbm/datasets/libero/`` (matches ``lbm.dataloader`` catalog)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import tensorflow_datasets as tfds
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

_SIMULATION_DIR = Path(__file__).resolve().parents[1]
if str(_SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_DIR))
from paths import datasets_dir  # noqa: E402

DEFAULT_REPO_ID = "libero"
DEFAULT_OUTPUT_DIR = datasets_dir() / DEFAULT_REPO_ID
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]


def main(
    data_dir: str | None = None,
    *,
    output_dir: Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    push_to_hub: bool = False,
    overwrite: bool = False,
) -> None:
    data_dir = data_dir or os.environ.get("LIBERO_RLDS")
    if not data_dir:
        raise SystemExit(
            "Pass --data-dir /path/to/modified_libero_rlds or set LIBERO_RLDS. "
            "Download: https://huggingface.co/datasets/openvla/modified_libero_rlds"
        )

    output_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    if output_path.exists():
        if output_path.is_symlink() and not overwrite:
            raise SystemExit(
                f"Refusing to overwrite symlink {output_path} -> {output_path.resolve()}. "
                "Pass --overwrite or --output-dir <new path>."
            )
        if not overwrite:
            raise SystemExit(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
        if output_path.is_symlink() or output_path.is_file():
            output_path.unlink()
        else:
            shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        root=output_path,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    print(f"Wrote LeRobot dataset → {dataset.root}")
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
