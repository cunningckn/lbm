"""LIBERO eval client — HTTP to an LBM policy server.

Install: bash simulation/libero/install_env.sh
Convert: bash simulation/libero/convert_libero_data_to_lerobot.sh /path/to/rlds

Terminal 1:
  bash simulation/libero/eval_policy.sh ./checkpoints/<run>/<step>.pt
Terminal 2:
  bash simulation/libero/eval_env.sh --task-suite-name libero_spatial
"""

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import sys

import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

_SIMULATION_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(_SIMULATION_DIR))
from client import PolicyClient  # noqa: E402
from paths import LBM_ROOT, libero_root  # noqa: E402

os.environ.setdefault("LIBERO_ROOT", str(libero_root()))

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    x = arr.astype(np.float32)
    if float(x.max()) <= 1.0:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "eval_results/libero/videos"  # Path to save videos

    seed: int = 7  # Simulator seed; policy RNG is configured separately.
    task_ids: tuple[int, ...] = ()  # Empty selects the full suite.
    init_offset: int = 0
    save_video: bool = True
    result_path: str = "eval_results/libero/result.json"


def eval_libero(args: Args) -> None:
    if args.replan_steps < 1 or args.num_trials_per_task < 1 or args.num_steps_wait < 0 or args.init_offset < 0:
        raise ValueError("replan/trial counts must be positive; wait/offset must be nonnegative")
    limits = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
    if args.task_suite_name not in limits:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = args.task_ids or tuple(range(suite.n_tasks))
    if len(set(task_ids)) != len(task_ids) or any(i < 0 or i >= suite.n_tasks for i in task_ids):
        raise ValueError("task_ids must be unique valid suite indices")
    initial_states = {i: suite.get_task_init_states(i) for i in task_ids}
    if any(args.init_offset + args.num_trials_per_task > len(states) for states in initial_states.values()):
        raise ValueError("trial range exceeds available official initial states")
    result_path = pathlib.Path(args.result_path)
    if not result_path.is_absolute():
        result_path = LBM_ROOT / result_path
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the report before contacting the server; never overwrite an earlier experiment.
    with result_path.open("x") as output:
        output.write("{}")
    video_out = pathlib.Path(args.video_out_path)
    if not video_out.is_absolute():
        video_out = LBM_ROOT / video_out
    report = dict(config=dataclasses.asdict(args), trials=[], errors=[], complete=False, successes=0)

    def save_report():
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(result_path)

    task_id = episode_idx = None
    save_report()
    try:
        if args.save_video:
            # Separate each run's videos, even when a caller reuses video_out_path.
            import tempfile

            video_out = pathlib.Path(tempfile.mkdtemp(prefix="rollouts-", dir=_video_directory(video_out)))
            report["video_directory"] = str(video_out)
        client = PolicyClient(args.host, args.port)
        report["metadata"] = client.get_server_metadata()
        for task_id in tqdm.tqdm(task_ids):
            task = suite.get_task(task_id)
            env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            try:
                for episode_idx in range(args.init_offset, args.init_offset + args.num_trials_per_task):
                    env.seed(args.seed + episode_idx)
                    env.reset()
                    obs = env.set_init_state(initial_states[task_id][episode_idx])
                    for _ in range(args.num_steps_wait):
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    plan = collections.deque()
                    images = []
                    done = False
                    requests = 0
                    for step in range(limits[args.task_suite_name]):
                        img = _to_uint8(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        if args.save_video:
                            images.append(img)
                        if not plan:
                            element = {
                                "state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ).astype(np.float32),
                                "images": {
                                    "image": img,
                                    "wrist_image": _to_uint8(
                                        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                                    ),
                                },
                                "prompt": str(description),
                                "timestamp": float(env.sim.data.time),
                                "episode_id": f"{task_id}:{episode_idx}",
                                "reset": requests == 0,
                            }
                            chunk = np.asarray(client.infer(element)["actions"])
                            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < args.replan_steps:
                                raise ValueError(
                                    f"expected at least {args.replan_steps} actions of width 7; got {chunk.shape}"
                                )
                            if not np.isfinite(chunk).all():
                                raise ValueError("nonfinite policy actions")
                            plan.extend(chunk[: args.replan_steps])
                            requests += 1
                        obs, _, done, _ = env.step(plan.popleft().tolist())
                        if done:
                            break
                    report["trials"].append(
                        dict(
                            task_id=task_id,
                            initial_state=episode_idx,
                            success=bool(done),
                            steps=step + 1,
                            requests=requests,
                        )
                    )
                    report["successes"] += int(bool(done))
                    save_report()
                    if args.save_video:
                        import imageio

                        imageio.mimwrite(video_out / f"task-{task_id}-trial-{episode_idx}.mp4", images, fps=20)
                    logging.info("Rollout: %s", report["trials"][-1])
            finally:
                env.close()
        report["complete"] = True
        save_report()
    except Exception as error:
        report["errors"].append(
            dict(type=type(error).__name__, message=str(error), task_id=task_id, initial_state=episode_idx)
        )
        save_report()
        raise
    logging.info("Successes: %s / %s", report["successes"], len(report["trials"]))


def _video_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    quat = np.array(quat, dtype=np.float64, copy=True)
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
