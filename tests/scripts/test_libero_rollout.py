import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def evaluator(monkeypatch):
    for name in ("libero", "libero.libero", "libero.libero.envs"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["libero.libero"].benchmark = types.SimpleNamespace()
    sys.modules["libero.libero"].get_libero_path = lambda name: "/unused"
    sys.modules["libero.libero.envs"].OffScreenRenderEnv = object
    path = Path(__file__).resolve().parents[2] / "simulation/libero/main.py"
    spec = importlib.util.spec_from_file_location("tested_libero_main", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def setup_rollout(module, monkeypatch, tmp_path, action=None, fail_reset=False):
    observations = []
    env = types.SimpleNamespace(closed=False, steps=0, sim=types.SimpleNamespace(data=types.SimpleNamespace(time=0.0)))
    obs = dict(
        agentview_image=np.zeros((2, 2, 3), dtype=np.uint8),
        robot0_eye_in_hand_image=np.zeros((2, 2, 3), dtype=np.uint8),
        robot0_eef_pos=np.zeros(3),
        robot0_eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        robot0_gripper_qpos=np.zeros(2),
    )

    def reset():
        if fail_reset:
            raise RuntimeError("reset failed")
        env.steps = 0
        env.sim.data.time = 0.0

    def step(action):
        env.steps += 1
        env.sim.data.time += 0.05
        return obs, 0, env.steps == 2, {}

    env.reset = reset
    env.seed = lambda seed: None
    env.set_init_state = lambda state: obs
    env.step = step
    env.close = lambda: setattr(env, "closed", True)
    suite = types.SimpleNamespace(n_tasks=1, get_task=lambda i: object(), get_task_init_states=lambda i: [0, 1])
    monkeypatch.setattr(
        module.benchmark, "get_benchmark_dict", lambda: {"libero_spatial": lambda: suite}, raising=False
    )
    monkeypatch.setattr(module, "_get_libero_env", lambda *args: (env, "task"))

    def infer(element):
        observations.append(element)
        return {"actions": np.zeros((1, 7)) if action is None else action}

    monkeypatch.setattr(
        module, "PolicyClient", lambda *args: types.SimpleNamespace(get_server_metadata=lambda: {}, infer=infer)
    )
    args = module.Args(
        num_trials_per_task=2,
        num_steps_wait=0,
        replan_steps=1,
        save_video=False,
        result_path=str(tmp_path / "result.json"),
    )
    return args, env, observations


def test_success_resets_history_and_closes_environment(evaluator, monkeypatch, tmp_path):
    args, env, requests = setup_rollout(evaluator, monkeypatch, tmp_path)
    evaluator.eval_libero(args)
    report = json.loads(Path(args.result_path).read_text())
    assert report["complete"] and report["successes"] == 2 and report["errors"] == []
    assert env.closed
    assert [r["reset"] for r in requests] == [True, False, True, False]
    assert [r["timestamp"] for r in requests] == [0.0, 0.05, 0.0, 0.05]
    assert [r["episode_id"] for r in requests] == ["0:0", "0:0", "0:1", "0:1"]
    with pytest.raises(FileExistsError):
        evaluator.eval_libero(args)


@pytest.mark.parametrize("action", [np.zeros((1, 6)), np.full((1, 7), np.nan), np.zeros((0, 7))])
def test_bad_actions_are_errors_not_failed_trials(evaluator, monkeypatch, tmp_path, action):
    args, env, _ = setup_rollout(evaluator, monkeypatch, tmp_path, action=action)
    with pytest.raises(ValueError):
        evaluator.eval_libero(args)
    report = json.loads(Path(args.result_path).read_text())
    assert not report["complete"] and report["trials"] == [] and len(report["errors"]) == 1
    assert env.closed


def test_reset_error_closes_environment(evaluator, monkeypatch, tmp_path):
    args, env, _ = setup_rollout(evaluator, monkeypatch, tmp_path, fail_reset=True)
    with pytest.raises(RuntimeError, match="reset failed"):
        evaluator.eval_libero(args)
    assert env.closed
    assert json.loads(Path(args.result_path).read_text())["errors"][0]["type"] == "RuntimeError"


def test_initial_state_range_checked_before_running(evaluator, monkeypatch, tmp_path):
    args, _, _ = setup_rollout(evaluator, monkeypatch, tmp_path)
    args.init_offset = 1
    with pytest.raises(ValueError, match="trial range"):
        evaluator.eval_libero(args)
    assert not Path(args.result_path).exists()
