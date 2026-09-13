"""Compare completed paired training runs without treating steps as independent seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    steps = [row['step'] for row in rows]
    if steps != sorted(set(steps)):
        raise ValueError(f'duplicate or out-of-order steps in {path}')
    return rows


def compare_pair(baseline, candidate, steps):
    baseline, candidate = Path(baseline), Path(candidate)
    contracts = [json.loads((root / f'completed-{steps}.json').read_text()) for root in (baseline, candidate)]
    if contracts[0]['variant'] != 'baseline':
        raise ValueError('first arm must be baseline')
    for key in ('seed', 'train_manifest', 'val_manifest', 'batch', 'validation_seed', 'steps'):
        if contracts[0][key] != contracts[1][key]:
            raise ValueError(f'paired contract differs: {key}')
    hashes = [(root / 'initial-weights.sha256').read_text().strip() for root in (baseline, candidate)]
    if not hashes[0] or hashes[0] != hashes[1]:
        raise ValueError('initial weights differ')
    losses = [read_rows(root / 'losses.jsonl') for root in (baseline, candidate)]
    for rows in losses:
        if [row['step'] for row in rows] != list(range(1, steps+1)):
            raise ValueError('training update sequence is incomplete')
        if not all(math.isfinite(row['loss']) for row in rows):
            raise ValueError('nonfinite training loss')
    if [row['rng_sha256'] for row in losses[0]] != [row['rng_sha256'] for row in losses[1]]:
        raise ValueError('training random sequences differ')
    validations = [read_rows(root / 'validation.jsonl') for root in (baseline, candidate)]
    if not validations[0] or [r['step'] for r in validations[0]] != [r['step'] for r in validations[1]]:
        raise ValueError('validation schedules differ or are empty')
    if validations[0][-1]['step'] != steps:
        raise ValueError('final validation is missing')
    curve = []
    for left, right in zip(*validations, strict=True):
        if set(left['sources']) != set(right['sources']):
            raise ValueError('validation sources differ')
        row = dict(step=left['step'], sources={})
        for name, ref in left['sources'].items():
            actual = right['sources'][name]
            if ref['count'] <= 0 or ref['count'] != actual['count']:
                raise ValueError('validation sample counts differ or are empty')
            x, y = ref['error_sum']/ref['count'], actual['error_sum']/actual['count']
            if not math.isfinite(x) or not math.isfinite(y) or min(x, y) < 0:
                raise ValueError('invalid validation error')
            relative = (y-x)/x if x else None
            row['sources'][name] = dict(baseline=x, candidate=y, absolute_change=y-x,
                                        relative_change=relative,
                                        review_required=abs(relative) > .05 if x else y != 0)
        curve.append(row)
    return dict(seed=contracts[0]['seed'], variant=contracts[1]['variant'],
                steps=steps, paired_random_sequences=True, validation=curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = compare_pair(args.baseline, args.candidate, args.steps)
    output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
