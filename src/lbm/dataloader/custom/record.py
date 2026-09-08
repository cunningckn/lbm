"""Episode index records for lazy custom IO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeRecord:
    """One episode on disk. Vectors/frames are read in ``__getitem__``."""

    kind: str
    path: str
    n_frames: int
    lang: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
