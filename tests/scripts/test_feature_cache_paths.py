from copy import deepcopy

import pytest

from lbm.feature_cache_paths import dataset_feature_path


def test_dataset_local_identity_tracks_inputs_not_runtime(tmp_path):
    metadata = dict(
        model={"history_length": 0},
        backbone_sha256="a",
        source_fingerprint="source-a",
        extraction={"dtype": "torch.bfloat16"},
        source_config={"num_workers": 2},
    )
    path = dataset_feature_path(tmp_path, "dino", metadata, "train")
    assert path.relative_to(tmp_path).parts[:3] == (".cache", "dino", "prebuilt")
    runtime = deepcopy(metadata)
    runtime["source_config"]["num_workers"] = 4
    assert dataset_feature_path(tmp_path, "dino", runtime, "train") == path
    for key, value in [
        ("model", {"history_length": 0.3}),
        ("backbone_sha256", "b"),
        ("source_fingerprint", "source-b"),
        ("extraction", {"dtype": "torch.float32"}),
    ]:
        changed = dict(metadata, **{key: value})
        assert dataset_feature_path(tmp_path, "dino", changed, "train") != path
    assert dataset_feature_path(tmp_path, "dino", metadata, "val") != path


@pytest.mark.parametrize("name", ["../outside", "/tmp/model", "..", "a/b"])
def test_model_name_cannot_escape_dataset_cache(tmp_path, name):
    with pytest.raises(ValueError):
        dataset_feature_path(tmp_path, name, {}, "train")


def test_reuse_checks_all_shards_and_contract(tmp_path):
    import torch

    from lbm.feature_cache_paths import verify_existing_cache
    from lbm.feature_shards import write_sharded_cache

    batch = dict(
        state=torch.zeros(2, 1),
        actions=torch.zeros(2, 1, 1),
        task_vec_clip=torch.zeros(2, 2),
        action_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        camera_mask=torch.ones(2, 1, dtype=torch.bool),
        vision_features={"image": torch.zeros(2, 1, 2)},
    )
    metadata = {"backbone_sha256": "weights", "source_config": {"num_workers": 2}}
    path = tmp_path / "cache"
    write_sharded_cache(path, lambda start: iter([batch]), rows=2, metadata=metadata, shard_rows=1)
    assert len(verify_existing_cache(path, dict(metadata, source_config={}), 2)) == 2
    with pytest.raises(ValueError, match="different"):
        verify_existing_cache(path, dict(metadata, backbone_sha256="changed"), 2)
    payload = next((path / "000001").glob("*.npy"))
    with payload.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        verify_existing_cache(path, metadata, 2)


def test_validation_rejects_different_extraction_but_allows_split():
    from lbm.training_features import FeatureDataset

    train, val = FeatureDataset.__new__(FeatureDataset), FeatureDataset.__new__(FeatureDataset)
    train.metadata = {"extraction": {"split": "train", "tokenizer_sha256": "same"}}
    val.metadata = {"extraction": {"split": "val", "tokenizer_sha256": "same"}}
    train.check_validation(val)
    val.metadata["extraction"]["tokenizer_sha256"] = "different"
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        train.check_validation(val)


def test_prepared_batch_keeps_history_ages_fp32():
    import torch

    from lbm.batch import cast_prepared_batch

    ages = torch.tensor([[0.0, -0.123456]], dtype=torch.float32)
    raw = dict(
        history_offsets=ages,
        state_history_offsets=ages,
        state=torch.ones(1, 2),
        vision_features={"image": torch.ones(1, 2)},
        history_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    cast = cast_prepared_batch(raw, torch.bfloat16)
    assert cast["state"].dtype == cast["vision_features"]["image"].dtype == torch.bfloat16
    assert cast["history_mask"].dtype == torch.bool
    assert cast["history_offsets"].dtype == cast["state_history_offsets"].dtype == torch.float32
    torch.testing.assert_close(cast["history_offsets"], ages, rtol=0, atol=0)


def test_mmap_shard_handle_limit_is_bounded_and_keeps_payload_file_backed(tmp_path):
    import numpy as np
    import torch

    from lbm.feature_shards import write_sharded_cache
    from lbm.training_features import FeatureDataset

    batch = dict(
        state=torch.zeros(4, 1),
        actions=torch.zeros(4, 1, 1),
        task_vec_clip=torch.zeros(4, 2),
        action_mask=torch.ones(4, 1, 1, dtype=torch.bool),
        camera_mask=torch.ones(4, 1, dtype=torch.bool),
        vision_features={"image": torch.zeros(4, 1, 2)},
    )
    path = tmp_path / "cache"
    write_sharded_cache(path, lambda start: iter([batch]), rows=4, metadata={}, shard_rows=1)
    cached = FeatureDataset(path, max_open_shards=3)
    for index in range(4):
        cached[index]
    assert len(cached._shards) == 3
    assert all(isinstance(array, np.memmap) for child in cached._shards.values() for array in child._arrays.values())
    child = cached._shards[3]
    cached[3]
    assert cached._shards[3] is child
    for invalid in (0, 65, True):
        with pytest.raises(ValueError, match="max_open_shards"):
            FeatureDataset(path, max_open_shards=invalid)
