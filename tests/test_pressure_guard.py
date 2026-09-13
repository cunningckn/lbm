"""The destructive stress entry point must reject unisolated environments."""
import os
from pathlib import Path

import pytest
from context.validation.audit.pressure_guard import check_isolation, record


@pytest.fixture
def isolated(tmp_path):
    group = tmp_path / 'test'
    group.mkdir()
    for name, value in {'memory.max': str(8 * 2**30), 'memory.swap.max': '0',
                        'memory.oom.group': '1', 'cgroup.procs': str(os.getpid())}.items():
        (group / name).write_text(value)
    membership = tmp_path / 'membership'
    membership.write_text('0::/test\n')
    return tmp_path, membership


def test_isolated_limit(isolated):
    root, membership = isolated
    assert check_isolation(root, membership) == (root / 'test', 8 * 2**30)


@pytest.mark.parametrize('name,value', [('memory.max', 'max'), ('memory.max', str(200 * 2**30)),
                                       ('memory.max', '0'), ('memory.swap.max', 'max'),
                                       ('memory.oom.group', '0'), ('cgroup.procs', '123\n456')])
def test_reject_unsafe_limits(isolated, name, value):
    root, membership = isolated
    (root / 'test' / name).write_text(value)
    with pytest.raises(RuntimeError):
        check_isolation(root, membership)


def test_reject_container_root(isolated):
    root, membership = isolated
    membership.write_text('0::/\n')
    with pytest.raises(RuntimeError, match='dedicated child'):
        check_isolation(root, membership)


def test_progress_survives_reopen(tmp_path):
    path = Path(tmp_path) / 'progress.jsonl'
    record(path, phase='copy')
    record(path, phase='read')
    assert path.read_text().splitlines() == ['{"phase": "copy"}', '{"phase": "read"}']
