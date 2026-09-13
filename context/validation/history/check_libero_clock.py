"""Run with the LIBERO environment and simulation/libero:simulation on PYTHONPATH."""

import json

import numpy as np
from client import decode_obs, encode_obs
from libero.libero import benchmark
from main import LIBERO_DUMMY_ACTION, _get_libero_env

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
env, _ = _get_libero_env(suite.get_task(0), 64, 123)
try:
    env.reset()
    before = float(env.sim.data.time)
    observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    after = float(env.sim.data.time)
    assert after > before
    payload = dict(
        state=np.zeros(8),
        images={"image": observation["agentview_image"]},
        timestamp=after,
        episode_id="clock-smoke",
    )
    assert decode_obs(encode_obs(payload))["timestamp"] == after
    print(json.dumps(dict(before=before, after=after, clock_and_codec_passed=True, policy_evaluated=False)))
finally:
    env.close()
