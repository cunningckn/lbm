"""Dataset-local paths for immutable frozen-encoder caches."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def cache_contract(metadata):
    # Runtime paths, logging and worker settings do not describe cached tensors.
    return {key: value for key, value in metadata.items() if key != "source_config"}


def dataset_feature_path(dataset_root, model_name, metadata, split):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_name) or split not in {"train", "val"}:
        raise ValueError("invalid cache model name or split")
    payload = json.dumps(cache_contract(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    return Path(dataset_root) / ".cache" / model_name / "prebuilt" / fingerprint / split


def verify_existing_cache(path, metadata, rows):
    from lbm.training_features import FeatureDataset

    cached = FeatureDataset(path)
    expected = json.loads(json.dumps(cache_contract(metadata)))
    if len(cached) != rows or cache_contract(cached.metadata) != expected:
        raise ValueError("existing feature cache has a different source/encoder/configuration")
    # Touch one row per shard to verify every payload checksum, with bounded mmap handles.
    if cached.manifest["version"] == 2:
        start = 0
        for shard in cached.manifest["shards"]:
            cached[start]
            start += shard["rows"]
    return cached
