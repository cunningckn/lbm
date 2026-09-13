"""Subtask intervals in physical episode coordinates; gaps are excluded from sampling."""
from __future__ import annotations

from dataclasses import replace

import numpy as np


def validate_segments(segments, n_frames):
    result = []
    last_stop = 0
    for row in segments:
        start, stop, text = row['start'], row['stop'], row['text']
        if (type(start) is not int or type(stop) is not int or
                not 0 <= start < stop <= n_frames or start < last_stop):
            raise ValueError('subtask intervals must be ordered, disjoint and within the physical episode')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('subtask instruction must be nonempty text')
        result.append(dict(start=start, stop=stop, text=text.strip()))
        last_stop = stop
    if not result:
        raise ValueError('subtask mode requires at least one annotated interval per selected episode')
    return result


def from_frame_tasks(indices, tasks):
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in 'iu' or not len(values):
        raise ValueError('frame task indices must be a nonempty integer vector')
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    return validate_segments([
        dict(start=int(a), stop=int(b), text=tasks.get(int(values[a]), ''))
        for a, b in zip(starts, stops, strict=True)
    ], len(values))


def with_subtasks(records, spec):
    from lbm.dataloader.custom.datasets import module_for

    reader = getattr(module_for(spec.name), 'read_subtasks', None)
    if reader is None:
        raise ValueError(f'{spec.name}: subtask annotations are not supported')
    result = []
    for record in records:
        segments = validate_segments(reader(record), record.n_frames)
        result.append(replace(record, extra={**record.extra, 'instruction_segments': segments}))
    return result


def sample_count(record):
    segments = record.extra.get('instruction_segments')
    return sum(row['stop']-row['start'] for row in segments) if segments else record.n_frames


def locate_frame(record, local):
    segments = record.extra.get('instruction_segments')
    if not segments:
        return local
    # Number of subtasks per physical episode is small; no frame-sized index allocation.
    for row in segments:
        length = row['stop']-row['start']
        if local < length:
            return row['start']+local
        local -= length
    raise IndexError('subtask sample index out of bounds')


def frame_instruction(record, frame):
    for row in record.extra.get('instruction_segments', []):
        if row['start'] <= frame < row['stop']:
            return row['start'], row['stop'], row['text']
    if record.extra.get('instruction_segments'):
        raise IndexError('frame is outside annotated subtask intervals')
    return 0, record.n_frames, record.lang
