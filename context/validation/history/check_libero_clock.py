"""Run in the LIBERO environment; set LIBERO_ROOT for a non-default source checkout."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "simulation"), str(ROOT / "simulation/libero")]

from paths import libero_root  # noqa: E402

sys.path.insert(0, str(libero_root()))

import numpy as np  # noqa: E402
from client import decode_obs, encode_obs  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from main import LIBERO_DUMMY_ACTION, _get_libero_env  # noqa: E402

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
