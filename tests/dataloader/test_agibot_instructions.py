"""Agibot task files contain independent per-episode annotations."""
import json

import pytest

from lbm.dataloader.custom.datasets import agibot
from lbm.dataloader.custom.scan_index import ScanIndexError, load_scan_index, save_scan_index


def test_episode_text_selection_and_old_scan_cache_rejection(tmp_path):
    info = tmp_path/'task_info/task_7.json'
    info.parent.mkdir()
    info.write_text(json.dumps([
        {'episode_id': 2, 'task_name': 'put cup away'},
        {'episode_id': 1, 'task_name': 'open drawer'},
    ]))
    pairs = []
    for episode in ('001', '2', '3'):
        path = tmp_path/'proprio_stats/7'/episode/'proprio_stats.h5'
        path.parent.mkdir(parents=True)
        path.touch()
        pairs.append((tmp_path, path))
    records = agibot._records(pairs, [3]*3, [{}]*3, agibot.SPEC)
    assert [r.lang for r in records] == ['open drawer', 'put cup away', '']
    dest = save_scan_index(tmp_path, agibot.SPEC, records)
    assert [r.lang for r in load_scan_index(tmp_path, agibot.SPEC)] == [r.lang for r in records]
    manifest = dest/'manifest.json'
    data = json.loads(manifest.read_text())
    data.pop('adapter_revision')
    manifest.write_text(json.dumps(data))
    with pytest.raises(ScanIndexError, match='adapter revision'):
        load_scan_index(tmp_path, agibot.SPEC)


def test_agibot_annotation_rejects_ambiguous_rows_and_preserves_legacy_dictionary(tmp_path):
    info = tmp_path/'task_info/task_7.json'
    info.parent.mkdir()
    info.write_text(json.dumps({'instruction': ' legacy text '}))
    assert agibot._instructions(tmp_path, '7') == {'*': 'legacy text'}
    for rows in ([{'task_name': 'missing id'}], [{'episode_id': 1}, {'episode_id': '1'}]):
        info.write_text(json.dumps(rows))
        with pytest.raises(ValueError):
            agibot._instructions(tmp_path, '7')
    info.write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        agibot._instructions(tmp_path, '7')


def test_annotation_edit_invalidates_cached_text(tmp_path):
    path = tmp_path/'proprio_stats/7/1/proprio_stats.h5'
    path.parent.mkdir(parents=True)
    path.touch()
    records = agibot._records([(tmp_path, path)], [3], [{}], agibot.SPEC)
    save_scan_index(tmp_path, agibot.SPEC, records)
    assert load_scan_index(tmp_path, agibot.SPEC)[0].lang == ''
    info = tmp_path/'task_info/task_7.json'
    info.parent.mkdir()
    info.write_text(json.dumps([{'episode_id': 1, 'task_name': 'new task'}]))
    with pytest.raises(ScanIndexError, match='annotation source changed'):
        load_scan_index(tmp_path, agibot.SPEC)
    records = agibot._records([(tmp_path, path)], [3], [{}], agibot.SPEC)
    save_scan_index(tmp_path, agibot.SPEC, records)
    assert load_scan_index(tmp_path, agibot.SPEC)[0].lang == 'new task'
    info.write_text(json.dumps([{'episode_id': 1, 'task_name': 'edited task'}]))
    with pytest.raises(ScanIndexError, match='annotation source changed'):
        load_scan_index(tmp_path, agibot.SPEC)
