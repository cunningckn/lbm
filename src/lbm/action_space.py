"""Action representation, packed-vector slices, and abs↔rel/delta conversion.

``rel`` is pose relative to the current state after gather. ``delta`` is
consecutive within the gathered chunk (first step vs state, later steps vs the
previous absolute pose). ``abs`` is the stored target. LIBERO eef is already
per-frame incremental on disk (``stored=True``); do not convert it.

Missing action labels are the next proprio at ``round(native_fps / action_freq)``
(always absolute). Relative groups are applied later by
:func:`actions_in_train_space`.
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

DEFAULT = "default"
XYZ_ROTVEC = "xyz_rotvec"
XYZ_QUAT = "xyz_quat"
XYZ_ROT6D = "xyz_rot6d"

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
_FMT = {
    "default": DEFAULT,
    "": DEFAULT,
    "xyz_rotvec": XYZ_ROTVEC,
    "xyz+rotvec": XYZ_ROTVEC,
    "rotvec": XYZ_ROTVEC,
    "xyz_quat": XYZ_QUAT,
    "xyz+quat": XYZ_QUAT,
    "quat": XYZ_QUAT,
    "xyz_rot6d": XYZ_ROT6D,
    "xyz+rot6d": XYZ_ROT6D,
    "rot6d": XYZ_ROT6D,
}
_POSE_WIDTH = {XYZ_ROTVEC: 6, XYZ_QUAT: 7, XYZ_ROT6D: 9}


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


def parse_format(value: str | None) -> str:
    if value is None:
        return DEFAULT
    return _parse(_FMT, str(value).strip().lower(), "format")


def pose_width(fmt: str) -> int:
    key = parse_format(fmt)
    try:
        return _POSE_WIDTH[key]
    except KeyError:
        raise ValueError(f"{fmt!r} is not a packed pose format") from None


def infer_format(kind: str, width: int) -> str:
    if parse_kind(kind) != EEF:
        return DEFAULT
    w = int(width)
    if w == 6:
        return XYZ_ROTVEC
    if w == 7:
        return XYZ_QUAT
    if w == 9:
        return XYZ_ROT6D
    return DEFAULT


def native_format(slices: tuple[ActionSlice, ...]) -> str:
    picked = next((sl for sl in slices if sl.kind == EEF), None)
    if picked is None:
        return DEFAULT
    return picked.format or infer_format(EEF, picked.width)


def eef_scale_mask(fmt: str) -> np.ndarray:
    """True = apply q01/q99. Rotation of rot6d/quat stays in ``[-1, 1]``."""
    key = parse_format(fmt)
    if key == XYZ_ROT6D:
        return np.array([True, True, True, False, False, False, False, False, False])
    if key == XYZ_QUAT:
        return np.array([True, True, True, False, False, False, False])
    if key == XYZ_ROTVEC:
        return np.array([True, True, True, True, True, True])
    return np.ones(0, dtype=bool)


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
    format: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rep", parse_rep(self.rep))
        object.__setattr__(self, "kind", parse_kind(self.kind))
        object.__setattr__(self, "norm", parse_norm(self.norm))
        object.__setattr__(self, "stored", bool(self.stored))
        fmt = str(self.format or "").strip()
        if not fmt:
            fmt = infer_format(self.kind, self.width)
        object.__setattr__(self, "format", parse_format(fmt))

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
            format=str(raw["format"]) if "format" in raw else "",
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
    format: str | None = None,
) -> tuple[ActionSlice, ...]:
    """Left/right EEF pose + gripper. ``eef`` is xyz+rot (6 / quat 7 / rot6d 9)."""
    i = 0
    out: list[ActionSlice] = []
    fmt = format or infer_format(EEF, eef)
    for side in sides:
        out.append(ActionSlice(f"{side}_eef", i, i + eef, parse_rep(rep), EEF, QUANTILE, format=fmt))
        i += eef
        out.append(ActionSlice(f"{side}_gripper", i, i + grip, ABS, GRIPPER, QUANTILE))
        i += grip
    return tuple(out)


def delta_eef(
    *,
    xyzrot: int = 6,
    grip: int = 1,
    state_grip: int = 7,
    format: str | None = None,
) -> tuple[ActionSlice, ...]:
    """Single EEF + gripper. xyz+rot is already per-frame incremental on disk."""
    fmt = format or infer_format(EEF, xyzrot)
    return (
        ActionSlice(
            "eef",
            0,
            xyzrot,
            DELTA,
            EEF,
            QUANTILE,
            state_start=0,
            state_end=xyzrot,
            stored=True,
            format=fmt,
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
    """True when some group is consecutive delta (not file-delta)."""
    return any(sl.rep == DELTA and not sl.stored for sl in slices)


def uses_rel(slices: tuple[ActionSlice, ...]) -> bool:
    """True when some group is relative to the current state over a horizon."""
    return any(sl.rep == REL and not sl.stored for sl in slices)


def packed_action_dim(slices: tuple[ActionSlice, ...], fallback: int = 0) -> int:
    if not slices:
        return int(fallback)
    return max(int(sl.end) for sl in slices)


def packed_state_dim(slices: tuple[ActionSlice, ...], fallback: int = 0) -> int:
    if not slices:
        return int(fallback)
    return max(int(sl.state_span()[1]) for sl in slices)


def _remap_slice(sl: ActionSlice, mode: str) -> ActionSlice:
    if sl.stored or sl.kind == GRIPPER:
        return sl
    if sl.rep == mode:
        return sl
    return replace(sl, rep=mode)


def resolve_action_space(
    spec,
    action_mode: str,
    action_kind: str | None = None,
    action_format: str | None = None,
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
    fmt = parse_format(action_format) if action_format else None
    if action_kind and parse_kind(action_kind) == EEF:
        from lbm.kinematics import remap_joint_slices_to_eef

        pose_fmt = fmt if fmt and fmt != DEFAULT else XYZ_ROTVEC
        space = remap_joint_slices_to_eef(
            space, embodiment=getattr(spec, "embodiment", None), pose_format=pose_fmt
        )
    elif fmt and fmt != DEFAULT:
        space = remap_eef_format(space, fmt)
    return space


def remap_eef_format(slices: tuple[ActionSlice, ...], action_format: str) -> tuple[ActionSlice, ...]:
    """Reindex packed EEF groups to ``action_format`` (widths 6 / 7 / 9)."""
    dst = parse_format(action_format)
    if dst == DEFAULT:
        return slices
    w_pose = pose_width(dst)
    act_reps: list[tuple[int, int, int]] = []
    st_reps: list[tuple[int, int, int]] = []
    for sl in slices:
        if sl.kind != EEF:
            continue
        src = sl.format or infer_format(EEF, sl.width)
        if src == dst or sl.width == w_pose:
            continue
        act_reps.append((sl.start, sl.end, w_pose))
        ss, se = sl.state_span()
        if se > ss:
            st_reps.append((ss, se, w_pose))
    if not act_reps and not st_reps:
        return tuple(replace(sl, format=dst) if sl.kind == EEF else sl for sl in slices)
    return tuple(_reindex_slice(sl, act_reps, st_reps, dst) for sl in slices)


def _map_index(old: int, replacements: list[tuple[int, int, int]]) -> int:
    shift = 0
    for start, end, new_w in sorted(replacements, key=lambda r: r[0]):
        if old <= start:
            return old + shift
        if old < end:
            return start + shift + (old - start)
        shift += int(new_w) - (end - start)
    return old + shift


def _reindex_slice(
    sl: ActionSlice,
    act_reps: list[tuple[int, int, int]],
    st_reps: list[tuple[int, int, int]],
    dst_format: str,
) -> ActionSlice:
    start = _map_index(sl.start, act_reps) if act_reps else sl.start
    end = _map_index(sl.end, act_reps) if act_reps else sl.end
    ss, se = sl.state_span()
    if se <= ss:
        new_ss, new_se = ss, se
    elif st_reps:
        new_ss = _map_index(ss, st_reps)
        new_se = _map_index(se, st_reps)
    else:
        new_ss, new_se = ss, se
    fmt = dst_format if sl.kind == EEF else sl.format
    return replace(
        sl,
        start=int(start),
        end=int(end),
        state_start=int(new_ss),
        state_end=int(new_se),
        format=fmt,
    )


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
    if ref.shape[0] == 1 and action.shape[0] != 1 and ref.ndim == action.ndim:
        ref = np.broadcast_to(ref, action.shape)
    return ref


def eef_relative(act, ref, fmt: str):
    from lbm.kinematics import invert44, matrix_to_pose, pose_to_matrix

    ta = pose_to_matrix(act, fmt)
    tr = pose_to_matrix(ref, fmt)
    if tr.ndim == 2 and ta.ndim == 3:
        tr = np.broadcast_to(tr, ta.shape)
    rel = invert44(tr) @ ta
    return matrix_to_pose(rel, fmt)


def eef_absolute(delta, ref, fmt: str):
    from lbm.kinematics import matrix_to_pose, pose_to_matrix

    td = pose_to_matrix(delta, fmt)
    tr = pose_to_matrix(ref, fmt)
    if tr.ndim == 2 and td.ndim == 3:
        tr = np.broadcast_to(tr, td.shape)
    return matrix_to_pose(tr @ td, fmt)


def geom_rel(act, ref, sl: ActionSlice):
    act = np.asarray(act, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if sl.kind == EEF:
        return eef_relative(act, ref, sl.format)
    ref = _broadcast_ref(act, ref)
    return act - ref


def geom_abs(delta, ref, sl: ActionSlice):
    delta = np.asarray(delta, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if sl.kind == EEF:
        return eef_absolute(delta, ref, sl.format)
    ref = _broadcast_ref(delta, ref)
    return delta + ref


def convert_pose(vec, src_format: str, dst_format: str) -> np.ndarray:
    from lbm.kinematics import matrix_to_pose, pose_to_matrix

    src = parse_format(src_format)
    dst = parse_format(dst_format)
    if src == dst:
        return np.asarray(vec, dtype=np.float32)
    return matrix_to_pose(pose_to_matrix(vec, src), dst)


def pack_to_format(
    state,
    action,
    slices: tuple[ActionSlice, ...],
    action_format: str | None,
) -> tuple[np.ndarray, np.ndarray, tuple[ActionSlice, ...]]:
    """Convert packed EEF groups to ``action_format``; joint/gripper copied."""
    st = np.asarray(state, dtype=np.float32)
    act = np.asarray(action, dtype=np.float32)
    dst = parse_format(action_format) if action_format else native_format(slices)
    if dst == DEFAULT or not slices:
        return st, act, slices
    squeezed = st.ndim == 1
    if squeezed:
        st, act = st[None], act[None]
    act_reps: list[tuple[int, int, np.ndarray]] = []
    st_reps: list[tuple[int, int, np.ndarray]] = []
    act_widths: list[tuple[int, int, int]] = []
    st_widths: list[tuple[int, int, int]] = []
    w_pose = pose_width(dst)
    for sl in slices:
        if sl.kind != EEF:
            continue
        src = sl.format or infer_format(EEF, sl.width)
        if src == dst:
            continue
        pose_act = convert_pose(act[..., sl.start : sl.end], src, dst)
        act_reps.append((sl.start, sl.end, pose_act))
        act_widths.append((sl.start, sl.end, w_pose))
        ss, se = sl.state_span()
        if se > ss:
            pose_st = convert_pose(st[..., ss:se], src, dst)
            st_reps.append((ss, se, pose_st))
            st_widths.append((ss, se, w_pose))
    if act_reps:
        from lbm.kinematics import splice_last

        act = splice_last(act, act_reps)
    if st_reps:
        from lbm.kinematics import splice_last

        st = splice_last(st, st_reps)
    if act_widths or st_widths:
        slices = tuple(_reindex_slice(sl, act_widths, st_widths, dst) for sl in slices)
    else:
        slices = tuple(replace(sl, format=dst) if sl.kind == EEF else sl for sl in slices)
    if squeezed:
        return st[0], act[0], slices
    return st, act, slices


def actions_in_train_space(
    action,
    state,
    slices: tuple[ActionSlice, ...],
    *,
    invert: bool = False,
):
    """Convert a gathered absolute chunk to (or from) the configured train space.

    ``action`` is ``(T, D)`` or ``(D,)``. ``state`` is the current proprio
    ``(Ds,)`` (or ``(T, Ds)`` for per-frame rel). Gripper and ``stored`` groups
    are left as-is. Computed ``delta`` is consecutive in the chunk; ``rel`` is
    every step minus the current state.
    """
    out = np.asarray(action, dtype=np.float32).copy()
    if not slices or out.size == 0:
        return out
    squeezed = out.ndim == 1
    if squeezed:
        out = out[None]
    ref_state = np.asarray(state, dtype=np.float32)
    t_len = int(out.shape[0])
    for sl in slices:
        if sl.stored or sl.kind == GRIPPER or sl.rep == ABS:
            continue
        piece = out[..., sl.start : sl.end]
        if sl.rep == REL:
            src = _ref_slice(ref_state, sl, sl.width)
            src = _broadcast_ref(piece, src)
            out[..., sl.start : sl.end] = geom_abs(piece, src, sl) if invert else geom_rel(piece, src, sl)
            continue
        if sl.rep != DELTA:
            continue
        if invert:
            cur = _ref_slice(ref_state if ref_state.ndim == 1 else ref_state[0], sl, sl.width)
            if cur.ndim > 1:
                cur = cur[0]
            acc = np.empty_like(piece)
            acc[0] = geom_abs(piece[0], cur, sl)
            for k in range(1, t_len):
                acc[k] = geom_abs(piece[k], acc[k - 1], sl)
            out[..., sl.start : sl.end] = acc
        else:
            cur = _ref_slice(ref_state if ref_state.ndim == 1 else ref_state[0], sl, sl.width)
            if cur.ndim > 1:
                cur = cur[0]
            acc = np.empty_like(piece)
            acc[0] = geom_rel(piece[0], cur, sl)
            if t_len > 1:
                acc[1:] = geom_rel(piece[1:], piece[:-1], sl)
            out[..., sl.start : sl.end] = acc
    return out[0] if squeezed else out


def apply_action_space(
    action,
    state,
    slices: tuple[ActionSlice, ...],
    *,
    invert: bool = False,
):
    """Convert packed absolute actions to (or from) the configured space."""
    return actions_in_train_space(action, state, slices, invert=invert)


def invert_action_space(action, state, slices: tuple[ActionSlice, ...]):
    return actions_in_train_space(action, state, slices, invert=True)


def column_scale_mask(slices: tuple[ActionSlice, ...], dim: int, *, field: str = "action") -> np.ndarray:
    """True where q01/q99 (or mean/std) should run. EEF rot6d/quat rotation is False."""
    mask = np.ones(int(dim), dtype=bool)
    for sl in slices:
        if field == "state":
            start, end = sl.state_span()
        else:
            start, end = int(sl.start), int(sl.end)
        if end <= start:
            continue
        start = max(0, start)
        end = min(int(dim), end)
        if end <= start:
            continue
        if sl.norm == NORM_NONE:
            mask[start:end] = False
        elif sl.kind == EEF:
            m = eef_scale_mask(sl.format)
            n = min(int(m.size), end - start)
            mask[start : start + n] = m[:n]
    return mask


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
