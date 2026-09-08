"""Smokes against real on-disk dumps. Set ``LBM_REAL_DATA=1`` to enable."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lbm.dataloader.custom import CUSTOM_SPECS, make_custom_dataset
from lbm.dataloader.custom.scan_index import cache_dir
from lbm.dataloader.paths import resolve_dataset
from tests.fixtures.custom_cfg import custom_cfg

_DIMS = {
    "kai0": (14, 14),
    "galaxea": (32, 14),
    "egoverse": (16, 16),
    "hy_lance": (16, 16),
    "abc": (14, 14),
    "hifi_umi": (20, 20),
    "das_gripper": (16, 16),
    "droid": (8, 8),
    "agibot": (20, 22),
    "libero": (8, 7),
    "rmbench": (14, 14),
}


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(_DIMS))
def test_real_dump_one_sample(name):
    if os.environ.get("LBM_REAL_DATA", "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip("set LBM_REAL_DATA=1 to smoke real dumps")
    root = resolve_dataset(name, required=False)
    if root is None or not Path(root).exists():
        pytest.skip(f"missing datasets/{name}")
    if not (cache_dir(root) / "manifest.json").is_file():
        pytest.skip(f"no scan index for {name}; run scripts/build_scan_index.sh first")
    state_dim, action_dim = _DIMS[name]
    ds = make_custom_dataset(root, name, data_cfg=custom_cfg(max_episodes=1))
    assert len(ds) >= 1
    sample = ds[0]
    assert sample["state"].shape[-1] == state_dim
    assert sample["action"].shape[-1] == action_dim
    assert np.isfinite(sample["state"]).all()
    assert np.isfinite(sample["action"]).all()
    assert len(sample["image"]) == len(CUSTOM_SPECS[name].camera_keys)
    live = [int(np.asarray(im).max()) for im in sample["image"]]
    # das_gripper cam_high is a black placeholder; wrists must still decode.
    if name == "das_gripper":
        assert max(live[1:]) > 0, live
    else:
        assert max(live) > 0, live
    print(
        f"{name}: n={len(ds)} state={tuple(sample['state'].shape)} "
        f"action={tuple(sample['action'].shape)} img_max={live} lang={str(sample['lang'])[:80]!r}"
    )
