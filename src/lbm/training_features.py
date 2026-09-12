"""Atomic, mmap-readable policy-input caches for frozen vision encoders."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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
        manifest = dict(version=1, rows=rows, fields=fields, metadata=metadata)
        (Path(tmp) / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
        stack.close()
        Path(tmp).rename(path)


class FeatureDataset(Dataset):
    def __init__(self, path):
        self.root = Path(path)
        raw = (self.root / 'manifest.json').read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get('version') != 1 or self.manifest.get('rows', 0) <= 0:
            raise ValueError('invalid feature cache manifest')
        self.metadata = self.manifest['metadata']
        self._arrays = None
        self._open()
        required = {'state', 'actions', 'task_vec_clip', 'action_mask', 'camera_mask'}
        if not required <= self._arrays.keys():
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
            arrays[key] = array
        self._arrays = arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = None
        return state

    def __len__(self):
        return self.manifest['rows']

    def __getitem__(self, index):
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
