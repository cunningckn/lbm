#!/usr/bin/env python3
"""Extract RMBench ``language_annotation.json`` into language subtask sidecars.

Reads frame-level subtask segments from ``third_party/rmbench/data/<task>/demo_clean/``
and writes normalized subtask spans under a LeRobot dataset root::

    <lerobot_root>/language/subtasks/
      manifest.json
      episodes/episode_XXXXXX.json

Episode indices follow the same global ordering as
:func:`convert_rmbench_data_to_lerobot.convert_tasks` (tasks appended in list order,
``episode_num`` episodes per task).

Usage (from openpi root)::

    uv run examples/rmbench/extract_language_annotations.py \\
      --lerobot-root ./data/rmbench

    uv run examples/rmbench/extract_language_annotations.py \\
      --lerobot-root ./data/rmbench_m1 --task-set m1 --episodes-per-task 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = EXAMPLE_DIR.parents[1]
DEFAULT_RMBENCH = OPENPI_ROOT / "third_party" / "rmbench"
DEFAULT_LEROBOT = OPENPI_ROOT / "data" / "rmbench"
SUBTASKS_DIRNAME = "subtasks"

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

M1_TASKS = [
    "observe_and_pickup",
    "put_back_block",
    "rearrange_blocks",
    "swap_blocks",
    "swap_T",
]

MN_TASKS = [
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "press_button",
    "place_block_mat",
]

TaskSet = Literal["all", "m1", "mn"]


def task_names_for_set(task_set: TaskSet) -> list[str]:
    if task_set == "m1":
        return list(M1_TASKS)
    if task_set == "mn":
        return list(MN_TASKS)
    return list(ALL_TASKS)


def raw_segments_to_spans(
    segments: list[list[Any]],
    *,
    fps: float,
    merge_adjacent: bool = True,
) -> list[dict[str, Any]]:
    """Convert RMBench ``[[text, frame_count], ...]`` to ``{text, start, end}`` spans."""
    t = 0.0
    spans: list[dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[0]).strip()
        try:
            n_frames = int(item[1])
        except (TypeError, ValueError):
            continue
        if not text or n_frames <= 0:
            continue
        start = t
        end = t + n_frames / fps
        if merge_adjacent and spans and spans[-1]["text"] == text:
            spans[-1]["end"] = end
        else:
            spans.append({"text": text, "start": start, "end": end})
        t = end
    return spans


def load_rmbench_fps(rmbench_root: Path, task: str, setting: str, default_fps: float) -> float:
    info_path = rmbench_root / "data" / task / setting / "meta" / "info.json"
    if info_path.is_file():
        try:
            return float(json.loads(info_path.read_text()).get("fps", default_fps))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return default_fps


def extract(
    *,
    rmbench_root: Path,
    lerobot_root: Path,
    task_names: list[str],
    setting: str,
    episodes_per_task: int,
    fps_default: float,
    merge_adjacent: bool,
) -> dict[str, Any]:
    out_dir = lerobot_root / "language" / SUBTASKS_DIRNAME
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source": "rmbench",
        "setting": setting,
        "episodes_per_task": episodes_per_task,
        "task_order": task_names,
        "merge_adjacent": merge_adjacent,
        "episodes": {},
    }

    global_idx = 0
    written = 0
    skipped = 0

    for task in task_names:
        ann_path = rmbench_root / "data" / task / setting / "language_annotation.json"
        if not ann_path.is_file():
            logger.warning("missing annotation file for task %s: %s", task, ann_path)
            skipped += episodes_per_task
            global_idx += episodes_per_task
            continue

        with ann_path.open(encoding="utf-8") as f:
            task_annotations = json.load(f)

        fps = load_rmbench_fps(rmbench_root, task, setting, fps_default)

        for local_idx in range(episodes_per_task):
            ep_key = f"episode_{local_idx}"
            segments = task_annotations.get(ep_key)
            subtasks: list[dict[str, Any]] = []
            if segments:
                subtasks = raw_segments_to_spans(segments, fps=fps, merge_adjacent=merge_adjacent)

            episode_doc = {
                "episode_index": global_idx,
                "task_name": task,
                "local_episode_index": local_idx,
                "fps": fps,
                "source_path": str(ann_path),
                "subtasks": subtasks,
                "segment_count_raw": len(segments) if segments else 0,
                "subtask_count": len(subtasks),
            }
            out_path = episodes_dir / f"episode_{global_idx:06d}.json"
            out_path.write_text(json.dumps(episode_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            manifest["episodes"][str(global_idx)] = {
                "task_name": task,
                "local_episode_index": local_idx,
                "fps": fps,
                "subtask_count": len(subtasks),
                "has_annotations": bool(subtasks),
            }
            if subtasks:
                written += 1
            else:
                skipped += 1
            global_idx += 1

    manifest["total_episodes"] = global_idx
    manifest["written_with_subtasks"] = written
    manifest["skipped_empty"] = skipped
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RMBench language_annotation → LeRobot import sidecars")
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=DEFAULT_LEROBOT,
        help=f"LeRobot dataset root (default: {DEFAULT_LEROBOT})",
    )
    parser.add_argument(
        "--rmbench-root",
        type=Path,
        default=DEFAULT_RMBENCH,
        help=f"RMBench repo root (default: {DEFAULT_RMBENCH})",
    )
    parser.add_argument(
        "--task-set",
        choices=("all", "m1", "mn"),
        default="all",
        help="Task split — must match convert_rmbench_data_to_lerobot task order",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task override (must match convert order if partial)",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="demo_clean",
        help="RMBench data setting (default: demo_clean)",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=50,
        help="Episodes per task — must match convert expert_data_num",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Fallback FPS when task meta/info.json is missing (LeRobot convert uses 50)",
    )
    parser.add_argument(
        "--no-merge-adjacent",
        action="store_true",
        help="Keep consecutive identical subtask segments separate",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_names = task_names_for_set(args.task_set)

    lerobot_root = args.lerobot_root.resolve()
    rmbench_root = args.rmbench_root.resolve()

    logger.info("tasks: %s", ", ".join(task_names))
    logger.info("rmbench: %s", rmbench_root)
    logger.info("output: %s/language/%s/", lerobot_root, SUBTASKS_DIRNAME)

    manifest = extract(
        rmbench_root=rmbench_root,
        lerobot_root=lerobot_root,
        task_names=task_names,
        setting=args.setting,
        episodes_per_task=args.episodes_per_task,
        fps_default=args.fps,
        merge_adjacent=not args.no_merge_adjacent,
    )
    logger.info(
        "done: %d episodes, %d with subtasks, %d empty → %s",
        manifest["total_episodes"],
        manifest["written_with_subtasks"],
        manifest["skipped_empty"],
        lerobot_root / "language" / SUBTASKS_DIRNAME,
    )


if __name__ == "__main__":
    main()
