"""Action representation, packed-vector slices, and abs↔rel conversion.

``delta`` (default) is the increment over the action window
(``a[t + hop] - a[t]``, hop from ``action_length`` × fps). ``rel`` subtracts
the current state after gather. ``abs`` is the stored target. LIBERO eef is
already per-frame incremental on disk (``stored=True``); do not convert it.

Missing action labels are the next proprio at
``round(native_fps / action_freq)`` (always absolute). Relative groups are
applied later by :func:`apply_action_space`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from lbm.temporal import native_stride


def _fit_dim(arr: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.shape[-1] == dim:
        return x
    out = np.zeros((*x.shape[:-1], dim), dtype=np.float32)
    n = min(dim, int(x.shape[-1]))
    out[..., :n] = x[..., :n]
    return out

ABS = "abs"
REL = "rel"
DELTA = "delta"

JOINT = "joint"
EEF = "eef"
GRIPPER = "gripper"

QUANTILE = "quantile"
MEAN_STD = "mean_std"
NORM_NONE = "none"

_REP = {
    "abs": ABS,
    "absolute": ABS,
    "rel": REL,
    "relative": REL,
    "delta": DELTA,
}
_KIND = {"joint": JOINT, "eef": EEF, "ee": EEF, "gripper": GRIPPER, "grip": GRIPPER}
_NORM = {
    "quantile": QUANTILE,
    "q01": QUANTILE,
    "mean_std": MEAN_STD,
    "meanstd": MEAN_STD,
    "none": NORM_NONE,
    "off": NORM_NONE,
    "identity": NORM_NONE,
}


def _parse(table: dict[str, str], value: str, what: str) -> str:
    key = str(value).strip().lower()
    try:
        return table[key]
    except KeyError:
        allowed = ", ".join(sorted(set(table.values())))
        raise ValueError(f"unknown {what} {value!r}; expected {allowed}") from None


def parse_rep(value: str) -> str:
    return _parse(_REP, value, "rep")


def parse_kind(value: str) -> str:
    return _parse(_KIND, value, "kind")


def parse_norm(value: str) -> str:
    return _parse(_NORM, value, "norm")


def is_relative(rep: str) -> bool:
    """True when packed actions should subtract (or add back) current state."""
    return parse_rep(rep) == REL


@dataclass(frozen=True)
class ActionSlice:
    name: str
    start: int
    end: int
    rep: str
    kind: str
    norm: str
    state_start: int | None = None
    state_end: int | None = None
    stored: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "rep", parse_rep(self.rep))
        object.__setattr__(self, "kind", parse_kind(self.kind))
        object.__setattr__(self, "norm", parse_norm(self.norm))
        object.__setattr__(self, "stored", bool(self.stored))

    @property
    def width(self) -> int:
        return int(self.end - self.start)

    def state_span(self) -> tuple[int, int]:
        if self.state_start is not None:
            start = int(self.state_start)
            end = int(self.state_end) if self.state_end is not None else start + self.width
            return start, end
        return int(self.start), int(self.end)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> ActionSlice:
        return cls(
            name=str(raw["name"]),
            start=int(raw["start"]),
            end=int(raw["end"]),
            rep=raw["rep"],
            kind=raw["kind"],
            norm=raw["norm"],
            state_start=raw["state_start"] if "state_start" in raw else None,
            state_end=raw["state_end"] if "state_end" in raw else None,
            stored=bool(raw["stored"]) if "stored" in raw else False,
        )


def unimanual_joint(*, arm: int = 7, grip: int = 1, rep: str = DELTA) -> tuple[ActionSlice, ...]:
    """Single arm + gripper, packed ``arm+grip`` (Franka / DROID)."""
    return (
        ActionSlice("arm", 0, arm, parse_rep(rep), JOINT, QUANTILE),
        ActionSlice("gripper", arm, arm + grip, ABS, GRIPPER, QUANTILE),
    )


def bimanual_joint(*, arm: int = 6, grip: int = 1, rep: str = DELTA) -> tuple[ActionSlice, ...]:
    """Left/right arm + gripper, packed ``arm+grip+arm+grip``."""
    i = 0
    out: list[ActionSlice] = []
    for side in ("left", "right"):
        out.append(ActionSlice(f"{side}_arm", i, i + arm, parse_rep(rep), JOINT, QUANTILE))
        i += arm
        out.append(ActionSlice(f"{side}_gripper", i, i + grip, ABS, GRIPPER, QUANTILE))
        i += grip
    return tuple(out)


def dual_eef(
    *,
    eef: int = 7,
    grip: int = 1,
    rep: str = DELTA,
    sides: tuple[str, str] = ("left", "right"),
) -> tuple[ActionSlice, ...]:
    """Left/right EEF pose + gripper. ``eef`` is xyz+rot (6 / quat 7 / rot6d 9)."""
    i = 0
    out: list[ActionSlice] = []
    for side in sides:
        out.append(ActionSlice(f"{side}_eef", i, i + eef, parse_rep(rep), EEF, QUANTILE))
        i += eef
        out.append(ActionSlice(f"{side}_gripper", i, i + grip, ABS, GRIPPER, QUANTILE))
        i += grip
    return tuple(out)


def delta_eef(*, xyzrot: int = 6, grip: int = 1, state_grip: int = 7) -> tuple[ActionSlice, ...]:
    """Single EEF + gripper. xyz+rot is already per-frame incremental on disk."""
    return (
        ActionSlice(
            "eef", 0, xyzrot, DELTA, EEF, QUANTILE, state_start=0, state_end=xyzrot, stored=True
        ),
        ActionSlice(
            "gripper",
            xyzrot,
            xyzrot + grip,
            ABS,
            GRIPPER,
            QUANTILE,
            state_start=state_grip,
            state_end=state_grip + grip,
        ),
    )


def stores_file_delta(slices: tuple[ActionSlice, ...]) -> bool:
    """True when some group is already incremental on disk (do not subsample)."""
    return any(sl.stored for sl in slices)


def uses_computed_delta(slices: tuple[ActionSlice, ...]) -> bool:
    """True when some group is ``a[t+hop] - a[t]`` over the action window."""
    return any(sl.rep == DELTA and not sl.stored for sl in slices)


def _remap_slice(sl: ActionSlice, mode: str) -> ActionSlice:
    if sl.stored or sl.kind == GRIPPER:
        return sl
    if sl.rep == mode:
        return sl
    return replace(sl, rep=mode)


def resolve_action_space(
    spec, action_mode: str, action_kind: str | None = None
) -> tuple[ActionSlice, ...]:
    mode = parse_rep(action_mode)
    space = tuple(spec.action_space)
    if space:
        space = tuple(_remap_slice(sl, mode) for sl in space)
    else:
        dim = int(spec.action_dim)
        if dim <= 0:
            raise ValueError(f"{spec.name}: empty action_space and action_dim={dim}")
        space = (ActionSlice("actions", 0, dim, mode, JOINT, QUANTILE),)
    if action_kind and parse_kind(action_kind) == EEF:
        from lbm.kinematics import remap_joint_slices_to_eef

        space = remap_joint_slices_to_eef(space, embodiment=getattr(spec, "embodiment", None))
    return space


def action_space_from_payload(raw: dict) -> tuple[ActionSlice, ...]:
    if "action_space" not in raw:
        return ()
    rows = raw["action_space"]
    if not rows:
        return ()
    return tuple(ActionSlice.from_dict(row) for row in rows)


def actions_from_state(
    state,
    *,
    native_fps: float,
    action_freq: float,
    mode: str,
) -> np.ndarray:
    """Per-frame actions from proprio when the dump has no action labels.

    ``stride = round(native_fps / action_freq)`` (at least 1). ``abs`` is
    ``state[t + stride]`` (last frames repeat). ``rel`` / ``delta`` is that
    minus ``state[t]``.
    """
    x = np.asarray(state, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    n = int(x.shape[0])
    if n == 0:
        return x
    stride = native_stride(native_fps, float(action_freq))
    nxt = x[np.minimum(np.arange(n) + stride, n - 1)]
    if parse_rep(mode) == ABS:
        return nxt
    return nxt - x


def _ref_slice(state: np.ndarray, sl: ActionSlice, width: int) -> np.ndarray:
    ss, se = sl.state_span()
    piece = state[ss:se] if state.ndim == 1 else state[..., ss:se]
    return _fit_dim(np.asarray(piece, dtype=np.float32), width)


def _broadcast_ref(action: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=np.float32)
    if ref.shape[-1] != action.shape[-1]:
        ref = _fit_dim(ref, int(action.shape[-1]))
    while ref.ndim < action.ndim:
        if ref.ndim == action.ndim - 1:
            ref = np.expand_dims(ref, axis=-2)
        else:
            ref = ref[None]
    return ref


def apply_delta_frames(action, slices: tuple[ActionSlice, ...], hop: int):
    """Per-frame computed delta: ``action[t + hop] - action[t]`` (not file-delta)."""
    act = np.asarray(action, dtype=np.float32)
    squeezed = act.ndim == 1
    if squeezed:
        act = act[None]
    if hop < 1 or act.shape[0] == 0 or not uses_computed_delta(slices):
        return act[0] if squeezed else act.copy()
    n = int(act.shape[0])
    nxt = act[np.minimum(np.arange(n) + int(hop), n - 1)]
    out = act.copy()
    for sl in slices:
        if sl.rep == DELTA and not sl.stored:
            out[:, sl.start : sl.end] = nxt[:, sl.start : sl.end] - act[:, sl.start : sl.end]
    return out[0] if squeezed else out


def apply_action_space(
    action,
    state,
    slices: tuple[ActionSlice, ...],
    *,
    invert: bool = False,
):
    """Convert packed absolute actions to (or from) the configured space.

    ``rel`` subtracts (or adds) the current state. Computed ``delta`` is inverted
    the same way at eval; the forward hop is :func:`apply_delta_frames`. File
    delta and ``abs`` stay as stored. ``action`` is ``(T, D)`` or ``(D,)``.
    """
    out = np.asarray(action, dtype=np.float32).copy()
    if not slices or out.size == 0:
        return out
    ref = np.asarray(state, dtype=np.float32)
    for sl in slices:
        add_state = is_relative(sl.rep) or (invert and sl.rep == DELTA and not sl.stored)
        if not add_state:
            continue
        piece = out[..., sl.start : sl.end]
        src = _broadcast_ref(piece, _ref_slice(ref, sl, sl.width))
        out[..., sl.start : sl.end] = piece + src if invert else piece - src
    return out


def invert_action_space(action, state, slices: tuple[ActionSlice, ...]):
    return apply_action_space(action, state, slices, invert=True)


def apply_action_space_frames(action, state, slices: tuple[ActionSlice, ...], *, hop: int = 1):
    """Per-frame computed delta (hop) then ``rel``: ``action[t] -= state[t]``."""
    act = apply_delta_frames(action, slices, hop)
    st = np.asarray(state, dtype=np.float32)
    if act.ndim == 1:
        return apply_action_space(act, st, slices)
    if st.ndim == 1:
        st = np.broadcast_to(st, (act.shape[0], st.shape[0]))
    n = min(act.shape[0], st.shape[0])
    out = np.asarray(act, dtype=np.float32).copy()
    out[:n] = apply_action_space(out[:n], st[:n], slices)
    return out


def derive_absolute_actions(
    state,
    spec,
    *,
    native_fps: float,
    action_freq: float,
) -> np.ndarray:
    """Next proprio at the action stride, packed to ``action_dim``."""
    st = np.asarray(state, dtype=np.float32)
    if st.ndim == 1:
        st = st[None]
    dim = int(spec.action_dim)
    slices = tuple(spec.action_space)
    if not slices:
        return _fit_dim(
            actions_from_state(st, native_fps=native_fps, action_freq=action_freq, mode=ABS),
            dim,
        )
    out = np.zeros((st.shape[0], dim), dtype=np.float32)
    for sl in slices:
        src = _ref_slice(st, sl, sl.width)
        if src.ndim == 1:
            src = src[None]
        nxt = actions_from_state(src, native_fps=native_fps, action_freq=action_freq, mode=ABS)
        out[:, sl.start : sl.end] = _fit_dim(nxt, sl.width)
    return out
