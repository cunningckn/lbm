"""Forward kinematics: joint angles → 6-D EEF (xyz + rotvec).

Lookup is ``(embodiment, n_joints)``, plus an optional arm ``side`` when left
and right chains differ. Unknown robots raise — do not relabel joints as EEF
without a model. Already-cartesian groups and extra DoF (head / waist / base)
are left unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from lbm.action_space import EEF, JOINT, ActionSlice, parse_kind


@dataclass(frozen=True)
class PoEChain:
    name: str
    slist: np.ndarray  # (6, n)
    m: np.ndarray  # (4, 4)


@dataclass(frozen=True)
class DHChain:
    name: str
    links: tuple[tuple[float, float, float], ...]  # (a, d, alpha)
    flange_d: float = 0.0
    modified: bool = False


@dataclass(frozen=True)
class SerialJoint:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


@dataclass(frozen=True)
class SerialChain:
    """URDF serial chain: fixed origin, then revolute about ``axis``."""

    name: str
    joints: tuple[SerialJoint, ...]
    ee_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


Chain = PoEChain | DHChain | SerialChain


def _joints(joints: np.ndarray, n_expected: int, name: str) -> np.ndarray:
    q = np.asarray(joints, dtype=np.float64)
    if q.ndim == 1:
        q = q[None]
    if q.shape[-1] != n_expected:
        raise ValueError(f"{name}: expected {n_expected} joints, got {q.shape[-1]}")
    return q


def _ident(n: int) -> np.ndarray:
    return np.broadcast_to(np.eye(4), (n, 4, 4)).copy()


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=np.float64)


def _exp6(s: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Space-form exponential. ``s`` is (6,), ``theta`` is (N,)."""
    n = int(theta.shape[0])
    w, v = s[:3], s[3:]
    wn = float(np.linalg.norm(w))
    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    if wn < 1e-12:
        out[:, :3, :3] = np.eye(3)
        out[:, :3, 3] = np.outer(theta, v)
        return out
    w = w / wn
    th = theta * wn
    what = _skew(w)
    what2 = what @ what
    eye = np.eye(3)
    sin = np.sin(th)[:, None, None]
    cos = np.cos(th)[:, None, None]
    rot = eye + sin * what + (1.0 - cos) * what2
    g = (
        (th[:, None, None] * eye)
        + ((1.0 - np.cos(th))[:, None, None] * what)
        + ((th - np.sin(th))[:, None, None] * what2)
    )
    out[:, :3, :3] = rot
    out[:, :3, 3] = (g @ v).reshape(n, 3)
    return out


def poe_fk(joints: np.ndarray, chain: PoEChain) -> np.ndarray:
    """``(N, n_joints)`` → ``(N, 4, 4)``."""
    q = _joints(joints, chain.slist.shape[1], chain.name)
    t = _ident(q.shape[0])
    for i in range(q.shape[1]):
        t = t @ _exp6(chain.slist[:, i], q[:, i])
    return t @ chain.m


def _dh_matrix(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    """Standard DH: RotZ(θ) TransZ(d) TransX(a) RotX(α)."""
    n = int(theta.shape[0])
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, 0, 0] = ct
    out[:, 0, 1] = -st * ca
    out[:, 0, 2] = st * sa
    out[:, 0, 3] = a * ct
    out[:, 1, 0] = st
    out[:, 1, 1] = ct * ca
    out[:, 1, 2] = -ct * sa
    out[:, 1, 3] = a * st
    out[:, 2, 1] = sa
    out[:, 2, 2] = ca
    out[:, 2, 3] = d
    out[:, 3, 3] = 1.0
    return out


def _mdh_matrix(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    """Modified DH: RotX(α) TransX(a) RotZ(θ) TransZ(d)."""
    n = int(theta.shape[0])
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, 0, 0] = ct
    out[:, 0, 1] = -st
    out[:, 0, 3] = a
    out[:, 1, 0] = st * ca
    out[:, 1, 1] = ct * ca
    out[:, 1, 2] = -sa
    out[:, 1, 3] = -sa * d
    out[:, 2, 0] = st * sa
    out[:, 2, 1] = ct * sa
    out[:, 2, 2] = ca
    out[:, 2, 3] = ca * d
    out[:, 3, 3] = 1.0
    return out


def dh_fk(joints: np.ndarray, chain: DHChain) -> np.ndarray:
    q = _joints(joints, len(chain.links), chain.name)
    link = _mdh_matrix if chain.modified else _dh_matrix
    t = _ident(q.shape[0])
    for i, (a, d, alpha) in enumerate(chain.links):
        t = t @ link(a, d, alpha, q[:, i])
    if chain.flange_d:
        t = t @ link(0.0, chain.flange_d, 0.0, np.zeros(q.shape[0]))
    return t


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF RPY: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _fixed44(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _rpy_matrix(rpy)
    out[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return out


def serial_fk(joints: np.ndarray, chain: SerialChain) -> np.ndarray:
    q = _joints(joints, len(chain.joints), chain.name)
    t = _ident(q.shape[0])
    for i, jnt in enumerate(chain.joints):
        axis = np.asarray(jnt.axis, dtype=np.float64)
        screw = np.concatenate([axis / np.linalg.norm(axis), np.zeros(3)])
        t = t @ _fixed44(jnt.xyz, jnt.rpy) @ _exp6(screw, q[:, i])
    return t @ _fixed44(chain.ee_xyz, chain.ee_rpy)


def rotmat_to_rotvec(rot: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` → ``(N, 3)`` axis-angle."""
    r = np.asarray(rot, dtype=np.float64)
    tr = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos)
    axis = np.stack(
        [r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]],
        axis=1,
    )
    small = angle < 1e-8
    near_pi = np.abs(angle - np.pi) < 1e-6
    out = np.zeros_like(axis)
    mid = ~small & ~near_pi
    scale = np.zeros(angle.shape[0], dtype=np.float64)
    scale[mid] = angle[mid] / (2.0 * np.sin(angle[mid]))
    out[mid] = axis[mid] * scale[mid, None]
    if np.any(near_pi):
        diag = np.stack([r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]], axis=1)
        ax = np.sqrt(np.clip((diag + 1.0) * 0.5, 0.0, 1.0))
        ax *= np.sign(axis)
        ax[np.linalg.norm(ax, axis=1) < 1e-8] = (1.0, 0.0, 0.0)
        ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
        out[near_pi] = ax[near_pi] * np.pi
    return out


def transforms_to_pose6(transforms: np.ndarray) -> np.ndarray:
    t = np.asarray(transforms, dtype=np.float64)
    return np.concatenate([t[:, :3, 3], rotmat_to_rotvec(t[:, :3, :3])], axis=1).astype(np.float32)


def _transforms(joints: np.ndarray, chain: Chain) -> np.ndarray:
    if isinstance(chain, PoEChain):
        return poe_fk(joints, chain)
    if isinstance(chain, DHChain):
        return dh_fk(joints, chain)
    return serial_fk(joints, chain)


# --- robot models ----------------------------------------------------------

_VX300S = PoEChain(
    "vx300s",
    slist=np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
        ],
        dtype=np.float64,
    ).T,
    m=np.array(
        [[1.0, 0.0, 0.0, 0.536494], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.42705], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
)

_FRANKA = DHChain(
    "franka",
    links=(
        (0.0, 0.333, 0.0),
        (0.0, 0.0, -np.pi / 2),
        (0.0, 0.316, np.pi / 2),
        (0.0825, 0.0, np.pi / 2),
        (-0.0825, 0.384, -np.pi / 2),
        (0.0, 0.0, np.pi / 2),
        (0.088, 0.0, np.pi / 2),
    ),
    flange_d=0.107,
    modified=True,
)

_YAM = PoEChain(
    "yam",
    slist=np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.1135, 0.0, 0.02],
            [0.0, -1.0, 0.0, 0.1135, 0.0, 0.244],
            [0.0, -1.0, 0.0, 0.173501, 0.0, -0.000998],
            [0.0, 0.0, -1.0, 0.0, 0.074998, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.173501, 0.0],
        ],
        dtype=np.float64,
    ).T,
    m=np.array(
        [[0.0, 0.0, -1.0, 0.110597], [0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.173502], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
)


def _g1_arm(side: str) -> SerialChain:
    """Genie-1 arm xacro. Right-arm recordings negate q2 relative to URDF."""
    j2 = {"left": 1.0, "right": -1.0}[side]
    return SerialChain(
        f"agibot_g1_{side}",
        joints=(
            SerialJoint((0.0, 0.0, 0.1859), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            SerialJoint((0.0, 0.0, 0.0), (-np.pi / 2, 0.0, np.pi / 2), (0.0, 0.0, j2)),
            SerialJoint((0.0, -0.305, 0.0), (-np.pi / 2, np.pi / 2, np.pi), (0.0, 0.0, 1.0)),
            SerialJoint((0.0, 0.0, 0.0), (-np.pi / 2, 0.0, np.pi), (0.0, 0.0, 1.0)),
            SerialJoint((0.0, -0.1975, 0.0), (-np.pi / 2, 0.0, -np.pi), (0.0, 0.0, 1.0)),
            SerialJoint((0.0, 0.0, 0.0), (-np.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0)),
            SerialJoint((0.0, -0.1805, 0.0), (np.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
    )


_G1_LEFT = _g1_arm("left")
_G1_RIGHT = _g1_arm("right")

_A1 = SerialChain(
    "galaxea_a1",
    joints=(
        SerialJoint((-0.0011147, 0.0, 0.0446), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        SerialJoint((0.0, 0.0, 0.1061), (np.pi / 2, 0.0, 0.0), (0.0, 0.0, -1.0)),
        SerialJoint((-0.34928, 0.019998, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        SerialJoint((0.02735, 0.069767, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        SerialJoint((0.2463, -0.00049894, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        SerialJoint((0.058249, 0.00050025, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ),
    ee_xyz=(0.1039, 0.0, 0.0),
    ee_rpy=(-np.pi / 2, 0.0, 0.0),
)

# (embodiment, n_joints, side). "" = not sided; "left"/"right" when the arms differ.
_CHAINS: dict[tuple[str, int, str], Chain] = {
    ("aloha", 6, ""): _VX300S,
    ("abc", 6, ""): _YAM,
    ("yam", 6, ""): _YAM,
    ("galaxea", 6, ""): _A1,
    ("oxe_droid", 7, ""): _FRANKA,
    ("franka", 7, ""): _FRANKA,
    ("agibot_genie1", 7, "left"): _G1_LEFT,
    ("agibot_genie1", 7, "right"): _G1_RIGHT,
    ("agibot", 7, "left"): _G1_LEFT,
    ("agibot", 7, "right"): _G1_RIGHT,
}

_FK_LOGGED: set[str] = set()


def _slice_side(name: str) -> str:
    key = str(name).lower()
    if key.startswith("right"):
        return "right"
    if key.startswith("left"):
        return "left"
    return ""


def _side_key(side: str | None) -> str:
    if side in (None, "", "left"):
        return "left"
    return str(side).lower()


def known_fk_embodiments() -> tuple[str, ...]:
    return tuple(sorted({emb for emb, _n, _side in _CHAINS}))


def has_chain(embodiment: str, n_joints: int) -> bool:
    emb, n = str(embodiment), int(n_joints)
    return any(e == emb and nj == n for e, nj, _side in _CHAINS)


def lookup_chain(embodiment: str, n_joints: int, *, side: str | None = None) -> Chain:
    emb, n = str(embodiment), int(n_joints)
    extra = _side_key(side)
    chain = _CHAINS.get((emb, n, extra)) or _CHAINS.get((emb, n, ""))
    if chain is None:
        known = ", ".join(sorted({f"{e}/{nj}j" for e, nj, _s in _CHAINS}))
        raise ValueError(f"no FK for embodiment {embodiment!r} with {n_joints} joints; registered: {known}")
    return chain


def joints_to_eef(joints: np.ndarray, embodiment: str, *, side: str | None = None) -> np.ndarray:
    """``(N, J)`` joint angles → ``(N, 6)`` xyz+rotvec."""
    q = np.asarray(joints, dtype=np.float64)
    squeezed = q.ndim == 1
    if squeezed:
        q = q[None]
    pose = transforms_to_pose6(_transforms(q, lookup_chain(embodiment, q.shape[-1], side=side)))
    return pose[0] if squeezed else pose


def _write_pose(dst: np.ndarray, start: int, end: int, pose: np.ndarray) -> None:
    width = int(end - start)
    if width < 6:
        raise ValueError(f"joint group width {width} < 6, cannot store xyz+rotvec")
    dst[..., start : start + 6] = pose
    if width > 6:
        dst[..., start + 6 : end] = 0.0


def remap_joint_slices_to_eef(
    slices: tuple[ActionSlice, ...],
    *,
    embodiment: str | None = None,
    names: frozenset[str] | None = None,
) -> tuple[ActionSlice, ...]:
    out: list[ActionSlice] = []
    for sl in slices:
        convert = sl.kind == JOINT and not sl.stored
        if convert and names is not None:
            convert = sl.name in names
        if convert and embodiment is not None:
            convert = has_chain(embodiment, sl.width)
        if not convert:
            out.append(sl)
            continue
        name = sl.name.replace("_arm", "_eef")
        if name == sl.name and sl.name == "arm":
            name = "eef"
        out.append(replace(sl, kind=EEF, name=name))
    return tuple(out)


def apply_joint_fk(
    state,
    action,
    spec,
    action_mode: str,
    action_kind: str | None,
    *,
    slices: tuple[ActionSlice, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[ActionSlice, ...]]:
    """If ``action_kind`` is eef, run FK on registered joint groups and relabel them."""
    from lbm.action_space import resolve_action_space

    kind = parse_kind(action_kind) if action_kind else None
    orig = slices if slices is not None else resolve_action_space(spec, action_mode)
    st = np.asarray(state, dtype=np.float32)
    act = np.asarray(action, dtype=np.float32)
    if kind in (None, JOINT):
        return st, act, orig
    groups = [sl for sl in orig if sl.kind == JOINT]
    if not groups:
        return st, act, orig
    arms = [sl for sl in groups if has_chain(spec.embodiment, sl.width)]
    if not arms:
        lookup_chain(spec.embodiment, groups[0].width)  # raises: no convertible arm
        return st, act, orig
    squeezed = st.ndim == 1
    if squeezed:
        st, act = st[None], act[None]
    st, act = st.copy(), act.copy()
    for sl in arms:
        side = _slice_side(sl.name)
        ss, se = sl.state_span()
        if 0 <= ss < se <= st.shape[-1]:
            _write_pose(st, ss, se, joints_to_eef(st[:, ss:se], spec.embodiment, side=side))
        _write_pose(act, sl.start, sl.end, joints_to_eef(act[:, sl.start : sl.end], spec.embodiment, side=side))
    labeled = remap_joint_slices_to_eef(orig, names=frozenset(sl.name for sl in arms))
    if spec.name not in _FK_LOGGED:
        _FK_LOGGED.add(spec.name)
        chain = lookup_chain(spec.embodiment, arms[0].width, side=_slice_side(arms[0].name))
        print(f"[data] {spec.name}: joint→eef via {chain.name} FK", flush=True)
    if squeezed:
        return st[0], act[0], labeled
    return st, act, labeled
