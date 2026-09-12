"""Resumable feature-cache shards; completed shards are immutable and checksummed."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _slice(batch, start, end):
    return {key: _slice(value, start, end) if isinstance(value, dict) else value[start:end]
            for key, value in batch.items()}


def write_sharded_cache(path, batches_factory, *, rows, metadata, shard_rows=1024):
    """batches_factory(start_row) skips completed rows without decoding them again."""
    from lbm.training_features import FeatureDataset, write_feature_cache

    path = Path(path)
    if rows <= 0 or shard_rows <= 0:
        raise ValueError('cache rows and shard_rows must be positive')
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + '.building')
    with path.with_name('.' + path.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise FileExistsError(f'feature cache already exists: {path}')
        staging.mkdir(exist_ok=True)
        contract = dict(rows=rows, metadata=metadata, shard_rows=shard_rows)
        contract = json.loads(json.dumps(contract, allow_nan=False))
        contract_path = staging / 'build.json'
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != contract:
                raise ValueError('cannot resume feature build with changed source/configuration')
        else:
            atomic_json(contract_path, contract)
        # Only unpublished temporary shard directories in this locked build belong to us.
        for partial in staging.glob('.[0-9][0-9][0-9][0-9][0-9][0-9].*'):
            if partial.is_dir() and not partial.is_symlink():
                shutil.rmtree(partial)
        shards, done = [], 0
        while (staging / f'{len(shards):06d}').exists():
            child = staging / f'{len(shards):06d}'
            cached = FeatureDataset(child, shard=True)
            if len(cached) != min(shard_rows, rows - done) or cached.metadata != contract['metadata']:
                raise ValueError('invalid completed feature shard')
            shards.append(dict(directory=child.name, rows=len(cached),
                               sha256=file_sha256(child / 'manifest.json')))
            done += len(cached)
        iterator = iter(batches_factory(done))
        pending, offset = None, 0
        while done < rows:
            count = min(shard_rows, rows - done)
            def shard_batches():
                nonlocal pending, offset
                left = count
                while left:
                    if pending is None:
                        try:
                            pending = next(iterator)
                        except StopIteration as exc:
                            raise ValueError('feature source ended before expected rows') from exc
                        offset = 0
                        if len(pending['state']) == 0:
                            raise ValueError('empty feature batch')
                    take = min(left, len(pending['state']) - offset)
                    yield _slice(pending, offset, offset + take)
                    offset += take
                    left -= take
                    if offset == len(pending['state']):
                        pending = None
            child = staging / f'{len(shards):06d}'
            write_feature_cache(child, shard_batches(), rows=count, metadata=metadata)
            shards.append(dict(directory=child.name, rows=count, sha256=file_sha256(child / 'manifest.json')))
            done += count
        if pending is not None or next(iterator, None) is not None:
            raise ValueError('feature source exceeds expected rows')
        atomic_json(staging / 'manifest.json', dict(version=2, rows=rows, metadata=metadata, shards=shards))
        staging.rename(path)
