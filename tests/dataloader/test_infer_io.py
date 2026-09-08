from types import SimpleNamespace

from lbm.batch import infer_policy_io, merge_policy_io
from lbm.dataloader.custom.datasets import CUSTOM_SPECS


def _ds(name: str, *, chunk: int):
    return SimpleNamespace(policy_io=CUSTOM_SPECS[name].as_policy_io(chunk_length=chunk))


def test_infer_single_dataset():
    spec = CUSTOM_SPECS["kai0"]
    io = infer_policy_io(_ds("kai0", chunk=50))
    assert io["camera_keys"] == spec.camera_keys
    assert io["action_dim"] == spec.action_dim
    assert io["state_dim"] == spec.state_dim
    assert io["chunk_length"] == 50


def test_merge_takes_union_and_max():
    kai0 = CUSTOM_SPECS["kai0"]
    libero = CUSTOM_SPECS["libero"]
    merged = merge_policy_io(
        [
            kai0.as_policy_io(chunk_length=50),
            libero.as_policy_io(chunk_length=10),
        ]
    )
    assert merged["action_dim"] == max(kai0.action_dim, libero.action_dim)
    assert merged["state_dim"] == max(kai0.state_dim, libero.state_dim)
    assert merged["chunk_length"] == 50
    assert merged["camera_keys"][0] == kai0.camera_keys[0]
    assert set(libero.camera_keys) <= set(merged["camera_keys"])


def test_infer_mixture_does_not_only_use_first():
    mix = SimpleNamespace(
        datasets=[
            _ds("kai0", chunk=50),
            _ds("libero", chunk=10),
        ]
    )
    io = infer_policy_io(mix)
    assert io["action_dim"] == CUSTOM_SPECS["kai0"].action_dim
    assert set(io["camera_keys"]) >= set(CUSTOM_SPECS["kai0"].camera_keys) | set(
        CUSTOM_SPECS["libero"].camera_keys
    )
