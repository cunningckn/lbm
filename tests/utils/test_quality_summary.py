import json

import pytest
from context.validation.correctness.summarize_quality import compare_rollouts, seed_summary


def test_rollout_pair_counts_disagreements_and_rejects_missing_official_state(tmp_path):
    paths = [tmp_path/'baseline.json', tmp_path/'combined.json']
    for path, successes in zip(paths, (set(range(30)), set(range(20, 40))), strict=True):
        report = dict(complete=True, errors=[], seed=7, independent_trials=True, suite='libero_spatial',
                      task_id=0, init_offset=0, replan_steps=5, max_steps=220, requested_trials=50,
                      task='task zero', successes=len(successes),
                      metadata=dict(policy_seed_base=2026, next_trial=0, inference_backend='eager', diffusion_steps=10),
                      trials=[dict(initial_state=i, steps=100 if i in successes else 220,
                                   success=i in successes) for i in range(50)])
        path.write_text(json.dumps(report))
    comparison = compare_rollouts(*paths)
    assert comparison['paired_outcomes'] == dict(both_success=10, baseline_only=20, candidate_only=10, both_failure=10)
    assert comparison['success_rate_difference'] == pytest.approx(-.2)
    report['trials'].pop()
    report['successes'] = sum(row['success'] for row in report['trials'])
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match='initial-state sequence'):
        compare_rollouts(*paths)
    with pytest.raises(ValueError, match='three finite'):
        seed_summary([-.2])
    summary = seed_summary([-.2, 0, .2])
    assert summary['mean'] == 0
    assert summary['t95_interval'][0] < -.2 < .2 < summary['t95_interval'][1]


def test_summary_uses_recorded_shortened_checkpoint_targets(tmp_path, monkeypatch):
    from context.validation.correctness import summarize_quality as module

    training = tmp_path/'training'
    training.mkdir()
    (training/'protocol-shortened.json').write_text(json.dumps({'training_steps': {'mixed': 3000, 'libero': 1500}}))
    called_steps = []

    def pair(baseline, candidate, steps):
        called_steps.append(steps)
        return dict(seed=int(baseline.name.split('-')[-1]), variant='combined', validation=[dict(sources={
            'source': dict(baseline=1., candidate=1., absolute_change=0., relative_change=0., review_required=False)})])

    monkeypatch.setattr(module, 'compare_pair', pair)
    monkeypatch.setattr(module, 'compare_rollouts', lambda *args: {'success_rate_difference': 0.})
    for seed in module.SEEDS:
        for variant in ('baseline', 'combined'):
            run = training/'libero'/f'{variant}-{seed}'
            rollout = tmp_path/'closed-loop'/run.name
            run.mkdir(parents=True)
            rollout.mkdir(parents=True)
            contract = json.dumps(dict(seed=seed, variant=variant, steps=1500))
            (run/'completed-1500.json').write_text(contract)
            (rollout/'training-contract.json').write_text(contract)
    result = module.summarize(tmp_path)
    assert result['training_steps'] == {'mixed': 3000, 'libero': 1500}
    assert called_steps == [3000]*3 + [1500]*3
