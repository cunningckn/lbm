from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

from lbm.serve import PolicyHTTPServer

_SIM = Path(__file__).resolve().parents[2] / "simulation"
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))
from client import PolicyClient  # noqa: E402


class _FakePolicy:
    def __init__(self) -> None:
        self.resets = 0
        self.last_obs = None

    def metadata(self) -> dict:
        return {
            "camera_keys": ["image", "wrist_image"],
            "state_dim": 8,
            "action_dim": 7,
            "chunk_length": 10,
            "history_size": 1,
            "diffusion_steps": 10,
            "embodiment_id": 25,
        }

    def reset(self) -> None:
        self.resets += 1

    def infer(self, obs: dict) -> np.ndarray:
        self.last_obs = obs
        return np.arange(70, dtype=np.float32).reshape(10, 7)


def _wait_for_server(client: PolicyClient, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            client._request("GET", "/health")
            return
        except Exception as exc:
            last_err = exc
            time.sleep(0.05)
    raise RuntimeError(f"policy server did not start: {last_err}")


def test_http_infer_roundtrip():
    policy = _FakePolicy()
    server = PolicyHTTPServer(("127.0.0.1", 0), policy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = PolicyClient(str(host), int(port), timeout=5)
        _wait_for_server(client)
        meta = client.get_server_metadata()
        assert meta["camera_keys"] == ["image", "wrist_image"]
        assert meta["action_dim"] == 7

        obs = {
            "state": np.zeros(8, dtype=np.float32),
            "images": {
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_image": np.full((4, 4, 3), 7, dtype=np.uint8),
            },
            "prompt": "open the drawer",
            "reset": True,
            "timestamp": 1720000000.125,
            "episode_id": "episode-3",
            "subtask_id": "reach",
        }
        out = client.infer(obs)
        assert out["actions"].shape == (10, 7)
        np.testing.assert_array_equal(out["actions"], np.arange(70, dtype=np.float32).reshape(10, 7))
        assert policy.last_obs["prompt"] == "open the drawer"
        assert policy.last_obs["reset"] is True
        for key in ("timestamp", "episode_id", "subtask_id"):
            assert policy.last_obs[key] == obs[key]
        np.testing.assert_array_equal(policy.last_obs["images"]["wrist_image"][0, 0], [7, 7, 7])

        client.reset()
        assert policy.resets == 1
    finally:
        server.shutdown()
        server.server_close()
