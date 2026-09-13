import json

import pytest
from context.validation.correctness.compare_training import compare_pair


def test_pair_rejects_rng_mismatch_and_incomplete_validation(tmp_path):
    arms = [tmp_path / 'baseline', tmp_path / 'combined']
    for arm in arms:
        arm.mkdir()
        contract = dict(variant=arm.name, seed=123, train_manifest='train', val_manifest='val',
                        batch=32, validation_seed=2026, steps=1)
        (arm / 'completed-1.json').write_text(json.dumps(contract))
        (arm / 'initial-weights.sha256').write_text('same-weights')
        (arm / 'losses.jsonl').write_text(json.dumps(dict(step=1, loss=1., rng_sha256='same-rng')))
        error = 100. if arm.name == 'baseline' else 110.
        (arm / 'validation.jsonl').write_text(json.dumps(dict(step=1, sources={
            'source': dict(error_sum=error, count=10)})))
    result = compare_pair(*arms, 1)
    change = result['validation'][0]['sources']['source']
    assert change['relative_change'] == pytest.approx(.1)
    assert change['review_required']
    (arms[1] / 'losses.jsonl').write_text(json.dumps(dict(step=1, loss=1., rng_sha256='different')))
    with pytest.raises(ValueError, match='random sequences'):
        compare_pair(*arms, 1)
    (arms[1] / 'losses.jsonl').write_text((arms[0] / 'losses.jsonl').read_text())
    (arms[1] / 'validation.jsonl').write_text('')
    with pytest.raises(ValueError, match='validation schedules'):
        compare_pair(*arms, 1)
