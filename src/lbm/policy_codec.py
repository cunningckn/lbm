"""JSON + base64 ndarray wire format for LBM policy HTTP infer.

Keep in sync with ``simulation/client.py`` (sim venvs cannot import ``lbm``).
"""

from __future__ import annotations

import base64
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
