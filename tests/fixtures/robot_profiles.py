"""Per-robot I/O specs from custom dump specs."""

from __future__ import annotations

from dataclasses import dataclass

from lbm.dataloader.custom.datasets import CUSTOM_SPECS


@dataclass(frozen=True)
class RobotIO:
    robot_type: str
    camera_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    embodiment: str
    fps: float | None = None
    state_dim: int | None = None
    action_dim: int | None = None


def robot_io(robot_type: str) -> RobotIO:
    spec = CUSTOM_SPECS[robot_type]
    return RobotIO(
        robot_type=robot_type,
        camera_keys=spec.camera_keys,
        state_keys=spec.state_columns,
        action_keys=spec.action_columns,
        embodiment=spec.embodiment,
        fps=spec.fps,
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
    )
