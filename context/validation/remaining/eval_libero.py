"""Bounded single-task LIBERO closed-loop evaluation against the standard HTTP server."""
import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import yaml

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--port', type=int, default=8123)
p.add_argument('--trials', type=int, default=10)
p.add_argument('--output', required=True)
a = p.parse_args()
repo = Path(a.repo)
root = repo / 'third_party/libero/libero/libero'
config = Path(a.output) / 'config'
config.mkdir(parents=True, exist_ok=True)
(config / 'config.yaml').write_text(yaml.safe_dump(dict(
    benchmark_root=str(root), bddl_files=str(root/'bddl_files'),
    init_states=str(root/'init_files'), assets=str(root/'assets'), datasets=str(repo/'datasets/libero'))))
os.environ['LIBERO_CONFIG_PATH'] = str(config)
sys.path.insert(0, str(repo/'third_party/libero'))
sys.path.insert(0, str(repo/'simulation'))
from client import PolicyClient  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

np.random.seed(7)
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
initial = suite.get_task_init_states(0)
client = PolicyClient(port=a.port)
metadata = client.get_server_metadata()
env = OffScreenRenderEnv(bddl_file_name=str(root/'bddl_files'/task.problem_folder/task.bddl_file),
                         camera_heights=256, camera_widths=256)
env.seed(7)
results = []
try:
    for trial in range(a.trials):
        env.reset()
        obs = env.set_init_state(initial[trial])
        for _ in range(10):
            obs, _, _, _ = env.step([0.]*6+[-1.])
        plan = deque()
        reset = True
        done = False
        for step in range(220):
            if not plan:
                q = obs['robot0_eef_quat'].copy()
                w = np.clip(q[3], -1., 1.)
                den = np.sqrt(1.-w*w)
                angle = np.zeros(3) if math.isclose(den, 0.) else q[:3]*2.*math.acos(w)/den
                state = np.concatenate([obs['robot0_eef_pos'], angle, obs['robot0_gripper_qpos']])
                element = dict(state=state.astype(np.float32), prompt=task.language, reset=reset,
                               images=dict(image=np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]),
                                           wrist_image=np.ascontiguousarray(
                                               obs['robot0_eye_in_hand_image'][::-1, ::-1])))
                chunk = client.infer(element)['actions']
                assert chunk.ndim == 2 and chunk.shape[1] == 7 and len(chunk) >= 5
                assert np.isfinite(chunk).all(), 'nonfinite policy actions'
                plan.extend(chunk[:5])
                reset = False
            obs, _, done, _ = env.step(plan.popleft().tolist())
            if done:
                break
        results.append(dict(trial=trial, steps=step+1, success=bool(done)))
        report = dict(task=task.language, seed=7, replan_steps=5, max_steps=220,
                      metadata=metadata, trials=results, successes=sum(r['success'] for r in results))
        (Path(a.output)/'result.json').write_text(json.dumps(report, indent=2))
        print('ROLLOUT', results[-1], flush=True)
finally:
    env.close()
print('CLOSED_LOOP ' + json.dumps(report), flush=True)
