"""Summarize completed paired experiments using training seeds as replicates."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from context.validation.correctness.compare_training import compare_pair

SEEDS = (123, 456, 789)


def seed_summary(values):
    """Descriptive Student-t interval over the protocol's three paired seeds."""
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError('expected three finite paired seed differences')
    mean = statistics.mean(values)
    margin = 4.302652729696142 * statistics.stdev(values) / math.sqrt(3)
    return dict(mean=mean, seed_values=values, t95_interval=[mean-margin, mean+margin],
                replicates=3, interval_scope='training seed variability; only three seeds')


def compare_rollouts(baseline, candidate):
    reports = [json.loads(Path(path).read_text()) for path in (baseline, candidate)]
    expected = dict(seed=7, independent_trials=True, suite='libero_spatial', task_id=0,
                    init_offset=0, replan_steps=5, max_steps=220, requested_trials=50)
    for report in reports:
        if report.get('complete') is not True or report.get('errors') != []:
            raise ValueError('closed-loop run is incomplete or has exceptions')
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError('closed-loop protocol differs')
        rows = report['trials']
        if [row['initial_state'] for row in rows] != list(range(50)):
            raise ValueError('official initial-state sequence differs')
        if any(type(row['success']) is not bool or not 1 <= row['steps'] <= 220 for row in rows):
            raise ValueError('invalid rollout outcome')
        if report['successes'] != sum(row['success'] for row in rows):
            raise ValueError('reported success total differs')
        metadata = report['metadata']
        for key, value in dict(policy_seed_base=2026, next_trial=0, inference_backend='eager',
                               diffusion_steps=10).items():
            if metadata.get(key) != value:
                raise ValueError('policy evaluation contract differs')
    if reports[0]['metadata'] != reports[1]['metadata'] or reports[0]['task'] != reports[1]['task']:
        raise ValueError('paired policies or tasks differ')
    outcomes = dict(both_success=0, baseline_only=0, candidate_only=0, both_failure=0)
    for left, right in zip(reports[0]['trials'], reports[1]['trials'], strict=True):
        key = ('both_success' if right['success'] else 'baseline_only') if left['success'] else (
            'candidate_only' if right['success'] else 'both_failure')
        outcomes[key] += 1
    return dict(baseline_successes=reports[0]['successes'], candidate_successes=reports[1]['successes'],
                trials=50, paired_outcomes=outcomes,
                success_rate_difference=(reports[1]['successes']-reports[0]['successes'])/50)


def summarize(root):
    root = Path(root)
    steps = {'mixed': 3000, 'libero': 3000}
    protocol = root/'training/protocol-shortened.json'
    if protocol.exists():
        steps = json.loads(protocol.read_text())['training_steps']
    if set(steps) != {'mixed', 'libero'} or any(type(n) is not int or n <= 0 for n in steps.values()):
        raise ValueError('invalid recorded training lengths')
    result = dict(complete=True, seeds=list(SEEDS), training_steps=steps, training={}, closed_loop={})
    for dataset in ('mixed', 'libero'):
        pairs = [compare_pair(root/'training'/dataset/f'baseline-{seed}',
                              root/'training'/dataset/f'combined-{seed}', steps[dataset]) for seed in SEEDS]
        if [pair['seed'] for pair in pairs] != list(SEEDS) or any(pair['variant'] != 'combined' for pair in pairs):
            raise ValueError('training seed or candidate differs')
        sources = set(pairs[0]['validation'][-1]['sources'])
        if any(set(pair['validation'][-1]['sources']) != sources for pair in pairs):
            raise ValueError('sources differ across training seeds')
        result['training'][dataset] = {}
        for source in sorted(sources):
            rows = [pair['validation'][-1]['sources'][source] for pair in pairs]
            relative = [row['relative_change'] for row in rows]
            result['training'][dataset][source] = dict(
                per_seed=rows, absolute_change=seed_summary([row['absolute_change'] for row in rows]),
                relative_change=seed_summary(relative) if all(v is not None for v in relative) else None,
                review_required=any(row['review_required'] for row in rows))
    rollouts = []
    for seed in SEEDS:
        directories = [root/'closed-loop'/f'{variant}-{seed}' for variant in ('baseline', 'combined')]
        for variant, directory in zip(('baseline', 'combined'), directories, strict=True):
            contract = json.loads((directory/'training-contract.json').read_text())
            marker = root/'training/libero'/f'{variant}-{seed}'/f"completed-{steps['libero']}.json"
            trained = json.loads(marker.read_text())
            if contract != trained:
                raise ValueError('rollout checkpoint contract differs from paired training')
        rollouts.append(dict(seed=seed, **compare_rollouts(*(directory/'result.json' for directory in directories))))
    result['closed_loop'] = dict(per_seed=rollouts,
                                 success_rate_difference=seed_summary([r['success_rate_difference'] for r in rollouts]),
                                 scope='Task 0, fixed official states 0–49; not an unseen-task or full-suite score')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    result = summarize(args.root)
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
