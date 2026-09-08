"""IO spec dataclass. Named dumps live under ``custom/datasets/<name>.py``."""

from __future__ import annotations

from dataclasses import dataclass

from lbm.action_space import ActionSlice
from lbm.dataloader.embodiment import embodiment_id_from_tag


@dataclass(frozen=True)
class CustomSpec:
    """Packed state/action + named cameras. One spec per robot dump."""

    name: str
    camera_keys: tuple[str, ...]
    state_dim: int
    action_dim: int
    fps: float
    embodiment_id: int
    embodiment: str
    state_columns: tuple[str, ...] = ("observation.state",)
    action_columns: tuple[str, ...] = ("action",)
    language_column: str = "task"
    image_size: int = 224
    kind: str = "auto"
    action_space: tuple[ActionSlice, ...] = ()

    def as_policy_io(self, *, chunk_length: int) -> dict:
        return {
            "camera_keys": self.camera_keys,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "video_keys": tuple(f"video.{cam}" for cam in self.camera_keys),
            "chunk_length": int(chunk_length),
            "embodiment_id": int(self.embodiment_id),
        }


def make_spec(
    name: str,
    embodiment: str,
    cameras: tuple[str, ...],
    state_dim: int,
    action_dim: int,
    fps: float,
    embodiment_id: int | None = None,
    **kwargs,
) -> CustomSpec:
    eid = int(embodiment_id) if embodiment_id is not None else embodiment_id_from_tag(embodiment)
    if eid >= 32:
        raise ValueError(f"{name}: embodiment_id {eid} must be < 32")
    return CustomSpec(
        name=name,
        camera_keys=cameras,
        state_dim=state_dim,
        action_dim=action_dim,
        fps=fps,
        embodiment_id=eid,
        embodiment=embodiment,
        **kwargs,
    )


WRISTS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def __getattr__(name: str):
    if name in {"CUSTOM_SPECS", "CUSTOM_MIXTURES", "uses_custom_backend"}:
        from lbm.dataloader.custom.datasets import CUSTOM_MIXTURES, CUSTOM_SPECS, uses_custom_backend

        mapping = {
            "CUSTOM_SPECS": CUSTOM_SPECS,
            "CUSTOM_MIXTURES": CUSTOM_MIXTURES,
            "uses_custom_backend": uses_custom_backend,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
