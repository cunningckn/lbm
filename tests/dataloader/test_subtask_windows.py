from dataclasses import replace

import numpy as np
import pytest

from lbm.dataloader.custom.dataset import CustomSingleDataset
from lbm.dataloader.custom.datasets import CUSTOM_SPECS
from lbm.dataloader.custom.instructions import from_frame_tasks, validate_segments
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.training_split import episode_view


def test_subtask_windows_exclude_gaps_and_clamp_to_physical_boundaries(monkeypatch):
    spec = CUSTOM_SPECS['libero']
    segments = validate_segments([dict(start=2, stop=5, text='first'),
                                  dict(start=7, stop=9, text='second')], 10)
    record = EpisodeRecord('demo', '/unused', 10, extra={'instruction_segments': segments})
    ds = CustomSingleDataset(spec, records=[record], action_mode='abs', image_size=2,
                             action_length=3/spec.fps, action_freq=spec.fps,
                             history_length=2/spec.fps, history_freq=spec.fps)
    state = np.zeros((10, spec.state_dim), np.float32)
    action = np.repeat(np.arange(10, dtype=np.float32)[:, None], spec.action_dim, axis=1)
    ds._policy_vectors = lambda _: (state, action, ds._space)
    monkeypatch.setattr('lbm.dataloader.custom.mmap_frames.camera_has_source', lambda *args: True)
    ds._read_record_frames = lambda record, cam, indices: np.broadcast_to(
        np.asarray(indices, dtype=np.uint8)[:, None, None, None], (len(indices), 2, 2, 3)).copy()
    assert len(ds) == 5
    assert [ds._locate(i) for i in range(5)] == [(0, 2), (0, 3), (0, 4), (0, 7), (0, 8)]
    assert ds[2]['lang'] == 'first'
    assert np.all(ds[2]['action'][:, 0] == 4)
    assert ds[3]['lang'] == 'second'
    assert np.all(ds[3]['image'][0] == 7)  # Never reads frames before the new subtask.
    from lbm.utils.preprocess import compute_norm_stats

    state[:] = np.arange(10, dtype=np.float32)[:, None]
    stats = compute_norm_stats(ds, progress=False)['norm_stats']['state']
    np.testing.assert_allclose(stats['mean'], 4.8)
    ds.records.append(replace(record))
    view = episode_view(ds, [1])
    assert len(view) == 5 and view._locate(4) == (0, 8)


def test_frame_tasks_preserve_transitions_and_reject_unknown_text_or_bad_intervals():
    rows = from_frame_tasks(np.array([2, 2, 5, 5, 2]), {2: 'pick', 5: 'place'})
    assert rows == [dict(start=0, stop=2, text='pick'), dict(start=2, stop=4, text='place'),
                    dict(start=4, stop=5, text='pick')]
    with pytest.raises(ValueError, match='nonempty'):
        from_frame_tasks(np.array([99]), {})
    for rows in ([dict(start=0, stop=3, text='a'), dict(start=2, stop=4, text='b')],
                 [dict(start=0, stop=11, text='a')], [dict(start=2, stop=2, text='a')]):
        with pytest.raises(ValueError):
            validate_segments(rows, 10)


def test_galaxea_uses_provided_english_subtask_text():
    from lbm.dataloader.custom.datasets.galaxea import _instruction_text

    assert _instruction_text('打开盖子@Open the cover.') == 'Open the cover.'
    assert _instruction_text('Move to marker @ A') == 'Move to marker @ A'
    assert _instruction_text('仅中文') == '仅中文'
