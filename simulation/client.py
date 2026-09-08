"""Stdlib HTTP client for the LBM policy server (sim venvs cannot import ``lbm``).

Sim venvs (LIBERO 3.10, RMBench 3.11) cannot load the LBM package (3.12 +
current torch). The server runs in the repo-root uv env; this client only
needs numpy + urllib.

Wire format matches ``lbm.policy_codec``.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np


def encode_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(obj: Any) -> np.ndarray:
    if isinstance(obj, dict) and "data" in obj and "shape" in obj:
        dtype = np.dtype(obj.get("dtype", "float32"))
        arr = np.frombuffer(base64.b64decode(obj["data"]), dtype=dtype)
        return np.reshape(arr, obj["shape"]).copy()
    return np.asarray(obj, dtype=np.float32)


def encode_obs(obs: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": encode_array(np.asarray(obs["state"], dtype=np.float32)),
        "images": {key: encode_array(np.asarray(image)) for key, image in obs["images"].items()},
        "prompt": str(obs.get("prompt") or ""),
    }
    if obs.get("reset"):
        payload["reset"] = True
    return payload


def decode_obs(payload: dict[str, Any]) -> dict[str, Any]:
    images = payload.get("images") or {}
    return {
        "state": decode_array(payload["state"]),
        "images": {key: decode_array(image) for key, image in images.items()},
        "prompt": str(payload.get("prompt") or ""),
        "reset": bool(payload.get("reset", False)),
    }


class PolicyClient:
    """POST ``{state, images, prompt}`` to ``scripts/serve_policy.py``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, timeout: float = 120.0):
        self.base = f"http://{host}:{int(port)}"
        self.timeout = float(timeout)
        # Sim machines often have http_proxy set; never proxy the local policy server.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach LBM policy server at {self.base}: {exc}") from exc
        return json.loads(body) if body else {}

    def get_server_metadata(self) -> dict[str, Any]:
        return self._request("GET", "/metadata")

    def reset(self) -> None:
        self._request("POST", "/reset", {})

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        raw = self._request("POST", "/infer", encode_obs(obs))
        if "error" in raw and "actions" not in raw:
            raise RuntimeError(raw["error"])
        return {"actions": decode_array(raw["actions"])}
