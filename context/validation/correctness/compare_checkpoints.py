"""Compare trusted training checkpoints bytewise, including replay and optimizer state."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


def compare_tree(left, right):
    differences = []
    tensors = 0

    def visit(a, b, path):
        nonlocal tensors
        if isinstance(a, torch.Tensor):
            tensors += 1
            if (not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape
                    or not torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))):
                differences.append(path)
        elif isinstance(a, np.ndarray):
            if (not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape
                    or a.tobytes() != b.tobytes()):
                differences.append(path)
        elif isinstance(a, Mapping):
            if not isinstance(b, Mapping) or set(a) != set(b):
                differences.append(path)
            else:
                for key in a:
                    visit(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                differences.append(path)
            else:
                for index, (x, y) in enumerate(zip(a, b, strict=True)):
                    visit(x, y, f'{path}.{index}')
        elif type(a) is not type(b) or a != b:
            differences.append(path)

    visit(left, right, 'checkpoint')
    return dict(tensors_compared=tensors, bitwise_equal=not differences, differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    baseline = torch.load(args.baseline, mmap=True, map_location='cpu', weights_only=False)
    candidate = torch.load(args.candidate, mmap=True, map_location='cpu', weights_only=False)
    report = compare_tree(baseline, candidate)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['bitwise_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
