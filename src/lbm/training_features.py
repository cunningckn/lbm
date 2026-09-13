"""Atomic, mmap-readable policy-input caches for frozen vision encoders."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

MODEL_FIELDS = (
    'camera_keys', 'state_dim', 'action_dim', 'action_length', 'action_freq',
    'history_length', 'history_freq', 'vision_encoder', 'vit_embed_dim',
    'vit_depth', 'vit_num_heads', 'task_embed_dim', 'language_encoder',
)


def backbone_fingerprint(backbone):
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _flatten(batch):
    fields = {}
    for key, value in batch.items():
        if key == 'images':
            continue
        if isinstance(value, dict):
            for subkey, tensor in value.items():
                fields[f'{key}/{subkey}'] = tensor
        elif torch.is_tensor(value):
            fields[key] = value
    return fields


def write_feature_cache(path, batches, *, rows, metadata):
    """Stream fixed-shape batches to .npy files, publishing only after all rows arrive."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'feature cache already exists: {path}')
    if rows <= 0:
        raise ValueError('feature cache requires positive rows')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{path.name}.', dir=path.parent) as tmp, ExitStack() as stack:
        fields, handles = {}, {}
        count = 0
        for batch in batches:
            flat = _flatten(batch)
            n = len(batch['state'])
            if not n or count + n > rows:
                raise ValueError('feature cache row count mismatch')
            if count and set(flat) != set(fields):
                raise ValueError('feature cache fields changed between batches')
            for key, tensor in flat.items():
                # Preserve BF16 bits in uint16 without doubling disk bandwidth.
                value = tensor.detach().cpu()
                if value.dtype == torch.bfloat16:
                    value = value.view(torch.uint16)
                array = value.numpy()
                if not count:
                    filename = f'{len(fields):03d}.npy'
                    shape = (rows, *array.shape[1:])
                    handle = stack.enter_context((Path(tmp) / filename).open('wb'))
                    np.lib.format.write_array_header_2_0(handle, dict(descr=array.dtype.str,
                                                                    fortran_order=False, shape=shape))
                    fields[key] = dict(file=filename, shape=list(shape), dtype=array.dtype.str,
                                       torch_dtype=str(tensor.dtype))
                    handles[key] = handle
                info = fields[key]
                if list(array.shape) != [n, *info['shape'][1:]] or array.dtype.str != info['dtype']:
                    raise ValueError(f'feature cache shape/dtype changed for {key}')
                handles[key].write(array.tobytes(order='C'))
            count += n
        if count != rows:
            raise ValueError(f'feature cache expected {rows} rows, got {count}')
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        from lbm.feature_shards import atomic_json, file_sha256

        for info in fields.values():
            info['sha256'] = file_sha256(Path(tmp) / info['file'])
        manifest = dict(version=1, rows=rows, fields=fields, metadata=metadata)
        atomic_json(Path(tmp) / 'manifest.json', manifest)
        stack.close()
        Path(tmp).rename(path)


class FeatureDataset(Dataset):
    def __init__(self, path, *, verify=True, shard=False):
        self.root = Path(path)
        raw = (self.root / 'manifest.json').read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get('version') not in (1, 2) or self.manifest.get('rows', 0) <= 0:
            raise ValueError('invalid feature cache manifest')
        self.metadata = self.manifest['metadata']
        groups = [] if shard else self.metadata.get('sources', [])
        end = 0
        for group in groups:
            name = group['name']
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('invalid feature source name')
            if group['start'] != end or not end < group['stop'] <= len(self):
                raise ValueError('invalid feature source range')
            end = group['stop']
        if groups and end != len(self):
            raise ValueError('feature sources do not cover all rows')
        self._arrays = None
        self._verify = verify
        self._shards = OrderedDict()
        self._verified = set()
        if self.manifest['version'] == 2:
            self._open_shards()
            return
        self._open()
        required = {'state', 'actions', 'task_vec_clip', 'action_mask', 'camera_mask'}
        if not required <= self._arrays.keys() or not any(k.startswith('vision_features/') for k in self._arrays):
            raise ValueError('feature cache is missing policy fields')

    def _open(self):
        arrays = {}
        for key, info in self.manifest['fields'].items():
            filename = info['file']
            if Path(filename).name != filename:
                raise ValueError('invalid feature cache filename')
            array = np.load(self.root / filename, mmap_mode='r', allow_pickle=False)
            if list(array.shape) != info['shape'] or array.dtype.str != info['dtype']:
                raise ValueError(f'invalid feature cache table: {key}')
            if len(array) != len(self):
                raise ValueError('inconsistent feature cache length')
            if self._verify and info.get('sha256'):
                from lbm.feature_shards import file_sha256

                if file_sha256(self.root / filename) != info['sha256']:
                    raise ValueError(f'feature cache checksum mismatch: {key}')
            arrays[key] = array
        self._arrays = arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = None
        state['_shards'] = OrderedDict()
        state['_verified'] = set()
        return state

    def __len__(self):
        return self.manifest['rows']

    def __getitem__(self, index):
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        if self.manifest['version'] == 2:
            shard = int(np.searchsorted(self._offsets, index, side='right'))
            start = int(self._offsets[shard - 1]) if shard else 0
            if shard not in self._shards:
                child = self.root / self.manifest['shards'][shard]['directory']
                self._shards[shard] = FeatureDataset(child, verify=shard not in self._verified, shard=True)
                self._verified.add(shard)
                if len(self._shards) > 2:
                    self._shards.popitem(last=False)
            self._shards.move_to_end(shard)
            return self._shards[shard][index - start]
        if self._arrays is None:
            self._open()
        out = {}
        for key, array in self._arrays.items():
            value = torch.from_numpy(np.array(array[index], copy=True))
            if self.manifest['fields'][key]['torch_dtype'] == 'torch.bfloat16':
                value = value.view(torch.bfloat16)
            if '/' in key:
                parent, child = key.split('/', 1)
                out.setdefault(parent, {})[child] = value
            else:
                out[key] = value
        return out

    def __getitems__(self, indices):
        """Read each shard once per batch, retaining sampler order and duplicates."""
        if self.manifest['version'] != 2:
            return [self[index] for index in indices]
        groups = {}
        for position, index in enumerate(indices):
            if not -len(self) <= index < len(self):
                raise IndexError(index)
            index %= len(self)
            shard = int(np.searchsorted(self._offsets, index, side='right'))
            groups.setdefault(shard, []).append((position, index))
        result = [None] * len(indices)
        for members in groups.values():
            for position, index in members:
                result[position] = self[index]
        return result

    def _open_shards(self):
        from lbm.feature_shards import file_sha256

        lengths, signature = [], None
        for shard in self.manifest['shards']:
            name = shard['directory']
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('invalid feature shard path')
            manifest = self.root / name / 'manifest.json'
            if file_sha256(manifest) != shard['sha256']:
                raise ValueError('feature shard manifest checksum mismatch')
            child = json.loads(manifest.read_text())
            fields = {k: (v['shape'][1:], v['dtype'], v['torch_dtype']) for k, v in child['fields'].items()}
            if child['metadata'] != self.metadata or child['rows'] != shard['rows'] or child['rows'] <= 0:
                raise ValueError('inconsistent feature shard metadata')
            if signature is not None and fields != signature:
                raise ValueError('inconsistent feature shard fields')
            signature = fields
            lengths.append(shard['rows'])
        if sum(lengths) != len(self):
            raise ValueError('inconsistent feature shard length')
        self._offsets = np.cumsum(lengths, dtype=np.int64)

    @property
    def datasets(self):
        from types import SimpleNamespace

        groups = self.metadata.get('sources', [])
        if not groups:
            return []
        result = []
        for group in groups:
            subset = Subset(self, range(group['start'], group['stop']))
            subset.spec = SimpleNamespace(name=group['name'])
            stops = [stop - group['start'] for start, stop in self.metadata.get('episode_ranges', [])
                     if group['start'] <= start < stop <= group['stop']]
            if stops:
                subset._episode_offsets = np.asarray(stops, dtype=np.int64)
            result.append(subset)
        return result

    @property
    def weights(self):
        return [group.get('weight', 1.0) for group in self.metadata.get('sources', [])]

    def apply_config(self, config):
        if config.model.train_vision_encoder or config.model.language_encoder != 'none':
            raise ValueError('feature caches require frozen vision and language_encoder=none')
        for key in MODEL_FIELDS:
            value = self.metadata['model'][key]
            setattr(config.model, key, tuple(value) if key == 'camera_keys' else value)
        if config.model.language_encoder != 'none':
            raise ValueError('feature cache language contract is unsupported')

    def check_backbone(self, model):
        if backbone_fingerprint(model.img_backbone) != self.metadata['backbone_sha256']:
            raise ValueError('feature cache was built with different encoder weights or precision')

    def check_validation(self, other):
        left, right = self.metadata.get('episode_ids'), other.metadata.get('episode_ids')
        if left is not None and right is not None:
            if set(left) & set(right):
                raise ValueError('training and validation feature episodes overlap')
        elif hasattr(self, 'root') and self.root.resolve() == other.root.resolve():
            raise ValueError('training and validation feature cache must differ')
        for key in ('model', 'normalization', 'action_spaces', 'backbone_sha256'):
            if self.metadata.get(key) != other.metadata.get(key):
                raise ValueError(f'training and validation feature cache {key} differ')

    def inference_normalization(self):
        norms = self.metadata.get('normalization', {})
        if len(norms) != 1:
            return None
        name, stats = next(iter(norms.items()))
        if stats is None:
            return None
        return dict(norm_stats=stats, action_space=self.metadata.get('action_spaces', {}).get(name, []))
