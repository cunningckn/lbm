"""FK against independent serial models and on-disk end poses."""

import numpy as np
import pytest

from lbm.dataloader.paths import datasets_root
from lbm.kinematics import SerialChain, SerialJoint, lookup_chain, poe_fk, serial_fk

_AGIBOT_H5 = datasets_root() / "agibot/AgiBotWorld_beta/proprio_stats/327/648642/proprio_stats.h5"
_GALAXEA_PQ = datasets_root() / "galaxea/Arrange_The_Fruits_20250822_013/data/chunk-000/episode_000000.parquet"

# Interbotix vx300s xacro: waist→wrist_rotate, then fixed to ee_gripper_link.
_VX300S_SERIAL = SerialChain(
    "vx300s_urdf",
    joints=(
        SerialJoint((0.0, 0.0, 0.079), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        SerialJoint((0.0, 0.0, 0.04805), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        SerialJoint((0.05955, 0.0, 0.3), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        SerialJoint((0.2, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        SerialJoint((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        SerialJoint((0.069744, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ),
    ee_xyz=(0.042825 + 0.025875 + 0.0385, 0.0, 0.0),
)

_YAM_SERIAL = SerialChain(
    "yam_urdf",
    joints=(
        SerialJoint((0.0, 0.0, 0.068), (0.0, np.pi / 2, np.pi), (-1.0, 0.0, 0.0)),
        SerialJoint((-0.0455, -0.0339, -0.02), (0.0, 0.0, -np.pi), (0.0, 1.0, 0.0)),
        SerialJoint((0.0, -0.0688, 0.264), (0.0, 0.0, -np.pi), (0.0, 1.0, 0.0)),
        SerialJoint((-0.0600003, -0.0688, -0.244999), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        SerialJoint((-0.0405003, 0.0338995, -0.0739989), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        SerialJoint((0.0404996, 0.0, -0.0356), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ),
)


def _rot_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.matmul(np.transpose(a, (0, 2, 1)), b)
    cos = np.clip((r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2] - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos)


def _pose_err(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    return dist, _rot_err(pred[:, :3, :3], gt[:, :3, :3])


def _constant_mount_residual(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arm-base FK vs robot-frame dump: ``gt ≈ T_mount @ pred`` with T_mount constant."""
    t0 = (gt @ np.linalg.inv(pred))[0]
    return _pose_err(pred, np.matmul(np.linalg.inv(t0), gt))


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*np.asarray(q).shape[:-1], 3, 3)


def _xyz_quat_to_t(xyz: np.ndarray, quat: np.ndarray) -> np.ndarray:
    t = np.zeros((*xyz.shape[:-1], 4, 4), dtype=np.float64)
    t[..., :3, :3] = _quat_xyzw_to_mat(quat)
    t[..., :3, 3] = xyz
    t[..., 3, 3] = 1.0
    return t


@pytest.mark.parametrize(
    "emb,serial",
    [("aloha", _VX300S_SERIAL), ("yam", _YAM_SERIAL)],
)
def test_poe_matches_urdf_serial(emb: str, serial: SerialChain):
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, size=(32, 6))
    q[0] = 0.0
    dist, ang = _pose_err(poe_fk(q, lookup_chain(emb, 6)), serial_fk(q, serial))
    assert dist.max() < 2e-3
    assert ang.max() < np.deg2rad(0.5)


def test_serial_fk_zero_is_composed_fixed_frames():
    from lbm.kinematics import _fixed44

    t = serial_fk(np.zeros(6), _YAM_SERIAL)[0]
    acc = np.eye(4)
    for jnt in _YAM_SERIAL.joints:
        acc = acc @ _fixed44(jnt.xyz, jnt.rpy)
    np.testing.assert_allclose(t, acc, atol=1e-8)


@pytest.mark.integration
def test_agibot_fk_vs_stored_end():
    if not _AGIBOT_H5.is_file():
        pytest.skip(f"missing {_AGIBOT_H5}")
    import h5py

    with h5py.File(_AGIBOT_H5, "r") as f:
        q = np.asarray(f["state/joint/position"], dtype=np.float64)
        xyz = np.asarray(f["state/end/position"], dtype=np.float64)
        quat = np.asarray(f["state/end/orientation"], dtype=np.float64)
    idx = np.linspace(0, len(q) - 1, num=min(400, len(q)), dtype=int)
    q, xyz, quat = q[idx], xyz[idx], quat[idx]
    samples = (
        ("right", q[:, 7:14], _xyz_quat_to_t(xyz[:, 1], quat[:, 1])),
        ("left", q[:, :7], _xyz_quat_to_t(xyz[:, 0], quat[:, 0])),
    )
    for side, joints, gt in samples:
        pred = serial_fk(joints, lookup_chain("agibot_genie1", 7, side=side))
        dist, ang = _constant_mount_residual(pred, gt)
        assert np.median(dist) * 1000 < 5.0, side
        assert np.median(ang) < np.deg2rad(1.0), side


@pytest.mark.integration
def test_galaxea_fk_vs_stored_ee_pose():
    if not _GALAXEA_PQ.is_file():
        pytest.skip(f"missing {_GALAXEA_PQ}")
    import pandas as pd

    df = pd.read_parquet(_GALAXEA_PQ)
    q = np.stack([np.asarray(v, dtype=np.float64).reshape(-1)[:6] for v in df["observation.state.left_arm"]])
    pose = np.stack([np.asarray(v, dtype=np.float64).reshape(-1)[:7] for v in df["observation.state.left_ee_pose"]])
    pred = serial_fk(q, lookup_chain("galaxea", 6))
    gt = _xyz_quat_to_t(pose[:, :3], pose[:, 3:7])
    dist, ang = _constant_mount_residual(pred, gt)
    assert np.median(dist) * 1000 < 2.0
    assert np.median(ang) < np.deg2rad(1.0)
