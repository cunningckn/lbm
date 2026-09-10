"""RMBench eval client — HTTP to an LBM policy server.

Terminal 1 — LBM checkpoint server (repo-root uv):
  bash simulation/rmbench/eval_policy.sh ./checkpoints/<run>/<step>.pt

Terminal 2:
  bash simulation/rmbench/eval_env.sh cover_blocks
"""

from __future__ import annotations

import collections
import dataclasses
import importlib
import json
import logging
import os
import pathlib
import sys

import cv2
import imageio
import numpy as np
import tqdm
import tyro
import yaml


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    replan_steps: int = 10

    #################################################################################################################
    # RMBench environment-specific parameters
    #################################################################################################################
    task_name: str = "cover_blocks"
    task_config: str = "demo_clean"
    instruction: str | None = None  # default: task_name with underscores → spaces
    num_trials: int = 50
    seed: int = 0

    #################################################################################################################
    # Utils
    #################################################################################################################
    eval_out_path: str = "eval_results/rmbench"
    rmbench_root: str | None = None
    # Project action_plan EEF/TCP trajectory onto replay videos (head camera).
    draw_action_plan_eef: bool = True


_SIMULATION_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_DIR))
from client import PolicyClient  # noqa: E402
from paths import LBM_ROOT  # noqa: E402
from paths import rmbench_root as _rmbench_root_fn  # noqa: E402

_RMBENCH_ROOT = pathlib.Path(os.environ.get("RMBENCH_ROOT", _rmbench_root_fn()))
# Match convert_rmbench_data_to_lerobot.py: native sim frames (e.g. 320x240) → 640x480 before model resize.
_DATASET_IMAGE_SIZE = (640, 480)
# Policy eval only needs these three RGB streams (see _camera_images).
_EVAL_CAMERA_NAMES = frozenset({"head_camera", "left_camera", "right_camera"})


def _default_instruction(rmbench_root: pathlib.Path, task_name: str, task_config: str) -> str:
    """Pick an instruction from raw demo JSON (matches training distribution when available)."""
    instr_dir = rmbench_root / "data" / task_name / task_config / "instructions"
    if instr_dir.is_dir():
        files = sorted(instr_dir.glob("episode*.json"))
        if files:
            with files[0].open("r", encoding="utf-8") as f:
                seen = json.load(f).get("seen", [])
            if seen:
                return str(np.random.choice(seen))
    return task_name.replace("_", " ")


def _trim_static_cameras(embodiment_config: dict) -> None:
    """Drop unused static cameras (e.g. front_camera) so they are not created or rendered."""
    static_cameras = embodiment_config.get("static_camera_list")
    if not static_cameras:
        return
    embodiment_config["static_camera_list"] = [
        cam for cam in static_cameras if cam.get("name") in _EVAL_CAMERA_NAMES
    ]


def _configure_eval_sensors(args: dict) -> None:
    """Disable sensors unused by LBM eval (depth, third view, pointcloud, etc.)."""
    data_type = args.setdefault("data_type", {})
    data_type["rgb"] = True
    data_type["qpos"] = True
    for key in ("third_view", "depth", "pointcloud", "observer", "endpose", "mesh_segmentation", "actor_segmentation"):
        data_type[key] = False

    args["collect_data"] = False
    args["eval_video_log"] = False
    args.pop("eval_video_save_dir", None)


def _load_task_args(rmbench_root: pathlib.Path, task_name: str, task_config: str) -> dict:
    with (rmbench_root / "task_config" / f"{task_config}.yml").open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    with (rmbench_root / "task_config" / "_embodiment_config.yml").open("r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    with (rmbench_root / "task_config" / "_camera_config.yml").open("r", encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    _configure_eval_sensors(args)

    embodiment_type = args["embodiment"]

    def embodiment_file(name: str) -> str:
        # Keep paths relative so they resolve after chdir(rmbench_root).
        path = embodiment_types[name]["file_path"]
        if path is None:
            raise ValueError(f"No embodiment file for {name}")
        return path

    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    def load_robot_config(robot_file: str) -> dict:
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    args["left_embodiment_config"] = load_robot_config(args["left_robot_file"])
    args["right_embodiment_config"] = load_robot_config(args["right_robot_file"])
    _trim_static_cameras(args["left_embodiment_config"])
    _trim_static_cameras(args["right_embodiment_config"])
    return args


def _make_env(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    return getattr(envs_module, task_name)()


def _resize_to_dataset(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, _DATASET_IMAGE_SIZE)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    x = arr.astype(np.float32)
    if float(x.max()) <= 1.0:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def _prepare_for_policy(img: np.ndarray) -> np.ndarray:
    return _to_uint8(_resize_to_dataset(img))


def _prepare_for_replay(img: np.ndarray) -> np.ndarray:
    return _to_uint8(_resize_to_dataset(img))


def _camera_images(observation: dict) -> dict[str, np.ndarray]:
    return {
        "cam_high": _prepare_for_policy(observation["observation"]["head_camera"]["rgb"]),
        "cam_right_wrist": _prepare_for_policy(observation["observation"]["right_camera"]["rgb"]),
        "cam_left_wrist": _prepare_for_policy(observation["observation"]["left_camera"]["rgb"]),
    }


def _project_world_to_pixel(
    point_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[float, float] | None:
    """Project a world-frame 3D point to head-camera pixel coordinates (OpenCV convention)."""
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    p_cam = extrinsic @ np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
    if p_cam[2] <= 1e-4:
        return None
    uv = intrinsic @ p_cam[:3]
    return float(uv[0] / uv[2]), float(uv[1] / uv[2])


def _scale_pixel_to_display(
    u: float,
    v: float,
    native_size: tuple[int, int],
    display_size: tuple[int, int],
) -> tuple[int, int]:
    native_w, native_h = native_size
    display_w, display_h = display_size
    return int(round(u * display_w / native_w)), int(round(v * display_h / native_h))


@dataclasses.dataclass
class _RobotQposMap:
    entity: object
    left_arm_indices: list[int]
    right_arm_indices: list[int]
    left_gripper_indices: list[tuple[int, float, float]]
    right_gripper_indices: list[tuple[int, float, float]]
    left_gripper_scale: tuple[float, float]
    right_gripper_scale: tuple[float, float]
    left_arm_dim: int
    is_dual_arm: bool


def _build_robot_qpos_map(env) -> _RobotQposMap:
    robot = env.robot
    entity = robot.left_entity
    active_joints = entity.get_active_joints()

    def _arm_indices(joints) -> list[int]:
        return [active_joints.index(j) for j in joints]

    def _gripper_indices(gripper_joints) -> list[tuple[int, float, float]]:
        return [(active_joints.index(entry[0]), entry[1], entry[2]) for entry in gripper_joints]

    return _RobotQposMap(
        entity=entity,
        left_arm_indices=_arm_indices(robot.left_arm_joints),
        right_arm_indices=_arm_indices(robot.right_arm_joints) if robot.is_dual_arm else [],
        left_gripper_indices=_gripper_indices(robot.left_gripper),
        right_gripper_indices=_gripper_indices(robot.right_gripper) if robot.is_dual_arm else [],
        left_gripper_scale=tuple(robot.left_gripper_scale),
        right_gripper_scale=tuple(robot.right_gripper_scale),
        left_arm_dim=len(robot.left_arm_joints),
        is_dual_arm=robot.is_dual_arm,
    )


def _apply_gripper_qpos(
    qpos: np.ndarray,
    gripper_val: float,
    gripper_scale: tuple[float, float],
    gripper_indices: list[tuple[int, float, float]],
) -> None:
    real = gripper_scale[0] + gripper_val * (gripper_scale[1] - gripper_scale[0])
    for idx, mimic, offset in gripper_indices:
        qpos[idx] = real * mimic + offset


def _eef_positions_from_qpos_action(
    env, qpos_map: _RobotQposMap, action: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """FK via temporary articulation qpos; matches env.get_arm_pose used in observations."""
    action = np.asarray(action, dtype=np.float64)
    entity = qpos_map.entity
    qpos = entity.get_qpos().copy()

    for i, idx in enumerate(qpos_map.left_arm_indices):
        qpos[idx] = action[i]
    _apply_gripper_qpos(
        qpos,
        action[qpos_map.left_arm_dim],
        qpos_map.left_gripper_scale,
        qpos_map.left_gripper_indices,
    )

    if qpos_map.is_dual_arm:
        offset = qpos_map.left_arm_dim + 1
        for i, idx in enumerate(qpos_map.right_arm_indices):
            qpos[idx] = action[offset + i]
        _apply_gripper_qpos(
            qpos,
            action[offset + len(qpos_map.right_arm_indices)],
            qpos_map.right_gripper_scale,
            qpos_map.right_gripper_indices,
        )

    saved_qpos = entity.get_qpos().copy()
    try:
        entity.set_qpos(qpos)
        # TCP aligns with the visible gripper on the arm better than the EE frame origin.
        left = np.asarray(env.robot.get_left_tcp_pose()[:3], dtype=np.float64)
        right = np.asarray(env.robot.get_right_tcp_pose()[:3], dtype=np.float64) if qpos_map.is_dual_arm else None
    finally:
        entity.set_qpos(saved_qpos)
    return left, right


def _draw_arm_eef_markers(
    img: np.ndarray,
    world_points: list[np.ndarray],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    native_size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw EEF action_plan markers directly on projected gripper locations."""
    prev: tuple[int, int] | None = None
    display_size = (img.shape[1], img.shape[0])
    for i, point in enumerate(world_points):
        pixel = _project_world_to_pixel(point, intrinsic, extrinsic)
        if pixel is None:
            prev = None
            continue
        u, v = _scale_pixel_to_display(pixel[0], pixel[1], native_size, display_size)
        if not (0 <= u < display_size[0] and 0 <= v < display_size[1]):
            prev = None
            continue

        if i == 0:
            cv2.drawMarker(
                img,
                (u, v),
                (0, 0, 0),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=18,
                thickness=3,
                line_type=cv2.LINE_AA,
            )
            cv2.drawMarker(
                img,
                (u, v),
                color,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=14,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        else:
            cv2.circle(img, (u, v), 7, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(img, (u, v), 5, color, -1, lineType=cv2.LINE_AA)
            if prev is not None:
                cv2.line(img, prev, (u, v), color, 2, lineType=cv2.LINE_AA)
        prev = (u, v)


def _overlay_action_plan_eef(
    img: np.ndarray,
    observation: dict,
    env,
    action_plan: collections.deque,
    qpos_map: _RobotQposMap,
) -> np.ndarray:
    """Project current + planned TCP positions onto the gripper locations in the head-camera frame."""
    head_cam = observation["observation"]["head_camera"]
    intrinsic = np.asarray(head_cam["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(head_cam["extrinsic_cv"], dtype=np.float64)
    native_rgb = head_cam["rgb"]
    native_size = (native_rgb.shape[1], native_rgb.shape[0])

    left_points: list[np.ndarray] = [np.asarray(env.robot.get_left_tcp_pose()[:3], dtype=np.float64)]
    right_points: list[np.ndarray] = []
    if qpos_map.is_dual_arm:
        right_points.append(np.asarray(env.robot.get_right_tcp_pose()[:3], dtype=np.float64))

    for action in action_plan:
        left_eef, right_eef = _eef_positions_from_qpos_action(env, qpos_map, np.asarray(action))
        if left_eef is not None:
            left_points.append(left_eef)
        if right_eef is not None:
            right_points.append(right_eef)

    if left_points:
        _draw_arm_eef_markers(img, left_points, intrinsic, extrinsic, native_size, (80, 220, 80))
    if right_points:
        _draw_arm_eef_markers(img, right_points, intrinsic, extrinsic, native_size, (60, 140, 255))
    return img


def _write_results_txt(
    path: pathlib.Path,
    *,
    args: Args,
    instruction: str,
    episode_rows: list[tuple[int, int, bool]],
    total_successes: int,
    done: bool,
) -> None:
    """Write / refresh success-rate summary under eval_out_path."""
    n = len(episode_rows)
    rate = (total_successes / n * 100.0) if n else 0.0
    lines = [
        "RMBench eval results",
        f"task_name: {args.task_name}",
        f"task_config: {args.task_config}",
        f"instruction: {instruction}",
        f"host: {args.host}:{args.port}",
        f"seed: {args.seed}",
        f"replan_steps: {args.replan_steps}",
        f"num_trials: {args.num_trials}",
        f"episodes_done: {n}",
        f"successes: {total_successes}",
        f"failures: {n - total_successes}",
        f"success_rate: {rate:.1f}% ({total_successes}/{n})",
        f"status: {'complete' if done else 'in_progress'}",
        "",
        "episode\tseed\tresult",
    ]
    for ep_id, ep_seed, succ in episode_rows:
        lines.append(f"{ep_id}\t{ep_seed}\t{'success' if succ else 'failure'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_rmbench(args: Args) -> None:
    np.random.seed(args.seed)
    rmbench_root = pathlib.Path(args.rmbench_root) if args.rmbench_root else _RMBENCH_ROOT
    if not (rmbench_root / "envs").is_dir():
        raise FileNotFoundError(f"RMBench not found at {rmbench_root}")

    eval_out = pathlib.Path(args.eval_out_path) / f"{args.task_name}"
    if not eval_out.is_absolute():
        eval_out = (LBM_ROOT / eval_out).resolve()
    eval_out.mkdir(parents=True, exist_ok=True)

    video_out = eval_out / "videos"
    video_out.mkdir(parents=True, exist_ok=True)
    results_txt = eval_out / "results.txt"

    # RMBench assets / relative embodiment paths resolve from its root.
    os.chdir(rmbench_root)
    if str(rmbench_root) not in sys.path:
        sys.path.insert(0, str(rmbench_root))

    task_args = _load_task_args(rmbench_root, args.task_name, args.task_config)
    instruction = args.instruction or _default_instruction(rmbench_root, args.task_name, args.task_config)

    client = PolicyClient(args.host, args.port)
    server_meta = client.get_server_metadata()
    logging.info("Connected to LBM policy server: %s", server_meta)

    env = _make_env(args.task_name)

    total_successes = 0
    episode_rows: list[tuple[int, int, bool]] = []
    seed = 100000 * (1 + args.seed)
    episode_id = 0
    pbar = tqdm.tqdm(total=args.num_trials, desc=args.task_name)

    while episode_id < args.num_trials:
        try:
            env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **task_args)
        except Exception as e:
            logging.warning("setup_demo failed seed=%s: %s", seed, e)
            try:
                env.close_env()
            except Exception:
                pass
            seed += 1
            continue

        env.set_instruction(instruction=instruction)
        episode_reset = True
        action_plan: collections.deque = collections.deque()
        replay_images: list[np.ndarray] = []
        qpos_map = _build_robot_qpos_map(env) if args.draw_action_plan_eef else None
        succ = False

        logging.info("Episode %d seed=%d instruction=%s", episode_id, seed, instruction)
        while env.take_action_cnt < env.step_lim:
            observation = env.get_obs()
            head_rgb = observation["observation"]["head_camera"]["rgb"]
            replay_frame = _prepare_for_replay(head_rgb)
            if args.draw_action_plan_eef and qpos_map is not None:
                replay_frame = _overlay_action_plan_eef(replay_frame, observation, env, action_plan, qpos_map)
            replay_images.append(replay_frame)

            images = _camera_images(observation)

            if not action_plan:
                element = {
                    "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
                    "images": images,
                    "prompt": instruction,
                }
                if episode_reset:
                    element["reset"] = True
                    episode_reset = False
                action_chunk = client.infer(element)["actions"]
                action_plan.extend(np.asarray(action_chunk)[: args.replan_steps])

            env.take_action(np.asarray(action_plan.popleft()))
            if env.eval_success:
                succ = True
                break

        if succ:
            total_successes += 1
        episode_rows.append((episode_id, seed, succ))
        logging.info("%s | %d/%d", "Success" if succ else "Fail", total_successes, episode_id + 1)

        suffix = "success" if succ else "failure"
        imageio.mimwrite(
            video_out / f"ep{episode_id:03d}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=10,
        )
        _write_results_txt(
            results_txt,
            args=args,
            instruction=instruction,
            episode_rows=episode_rows,
            total_successes=total_successes,
            done=False,
        )

        env.close_env(clear_cache=((episode_id + 1) % int(task_args.get("clear_cache_freq", 5)) == 0))
        episode_id += 1
        seed += 1
        pbar.update(1)
        pbar.set_postfix(success_rate=f"{total_successes / episode_id * 100:.1f}%")

    pbar.close()
    _write_results_txt(
        results_txt,
        args=args,
        instruction=instruction,
        episode_rows=episode_rows,
        total_successes=total_successes,
        done=True,
    )
    logging.info(
        "Total success rate: %.1f%% (%d/%d)",
        total_successes / args.num_trials * 100,
        total_successes,
        args.num_trials,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    eval_rmbench(tyro.cli(Args))
