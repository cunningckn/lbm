"""Projector indices for robot embodiments. Independent of any dataset backend.

Ids must stay below 32 (action-expert embedding table).
"""

from __future__ import annotations

EMBODIMENT_IDS: dict[str, int] = {
    "new_embodiment": 31,
    "aloha": 7,
    "ur5": 8,
    "arx5": 9,
    "dos-w1": 10,
    "galaxea": 11,
    "egoverse": 12,
    "das_gripper": 13,
    "hy_lance": 14,
    "hifi_umi": 16,
    "oxe_droid": 17,
    "oxe_bridge": 18,
    "oxe_rt1": 19,
    "abc": 20,
    "gr1": 24,
    "franka": 25,
    "agibot_genie1": 26,
}

DEFAULT_EMBODIMENT_ID = EMBODIMENT_IDS["new_embodiment"]

if any(int(v) >= 32 for v in EMBODIMENT_IDS.values()):
    raise ValueError("embodiment ids must be < 32")


def embodiment_id_from_tag(tag: str | None) -> int:
    if not tag:
        return DEFAULT_EMBODIMENT_ID
    return int(EMBODIMENT_IDS.get(str(tag), DEFAULT_EMBODIMENT_ID))
