"""stderr tqdm that stays on when stdout is piped; silent under pytest / workers."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def progress_enabled() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if os.environ.get("LBM_NO_PROGRESS"):
        return False
    disable = os.environ.get("TQDM_DISABLE", "")
    if disable and disable not in {"0", "false", "False"}:
        return False
    try:
        from torch.utils.data import get_worker_info

        if get_worker_info() is not None:
            return False
    except ImportError:
        pass
    return True


class _NoBar:
    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


def progress_bar(
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    **kwargs,
):
    if not progress_enabled():
        return _NoBar()
    try:
        from tqdm import tqdm
    except ImportError:
        return _NoBar()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        file=sys.stderr,
        mininterval=0.3,
        **kwargs,
    )


def track(
    iterable: Iterable[T],
    *,
    desc: str = "",
    total: int | None = None,
    unit: str = "it",
    leave: bool = True,
    **kwargs,
) -> Iterable[T]:
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        file=sys.stderr,
        mininterval=0.3,
        disable=not progress_enabled(),
        **kwargs,
    )
