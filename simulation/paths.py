"""Layout helpers for simulation convert / eval scripts.

Repo layout::

    lbm/
      datasets/          # LeRobot dumps (libero, rmbench, …)
      simulation/lerobot # uv env for convert
      simulation/libero
      simulation/rmbench
      third_party/libero
      third_party/rmbench
"""

from __future__ import annotations

import os
from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parent
LBM_ROOT = SIMULATION_DIR.parent
THIRD_PARTY = LBM_ROOT / "third_party"
DATASETS = LBM_ROOT / "datasets"
LEROBOT_DIR = SIMULATION_DIR / "lerobot"
LIBERO_ROOT = THIRD_PARTY / "libero"
RMBENCH_ROOT = THIRD_PARTY / "rmbench"


def datasets_dir() -> Path:
    env = os.environ.get("LBM_DATASETS")
    if env:
        return Path(env).expanduser().resolve()
    return DATASETS


def rmbench_root() -> Path:
    env = os.environ.get("RMBENCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return RMBENCH_ROOT


def libero_root() -> Path:
    env = os.environ.get("LIBERO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return LIBERO_ROOT
