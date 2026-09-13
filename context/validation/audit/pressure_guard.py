"""Fail closed unless a pressure run has its own bounded cgroup v2."""
import json
import os
from pathlib import Path


def check_isolation(cgroup_root=Path('/sys/fs/cgroup'), membership=Path('/proc/self/cgroup')):
    entries = membership.read_text().splitlines()
    relative = next((line[3:] for line in entries if line.startswith('0::')), '/')
    if relative == '/' or '..' in Path(relative).parts:
        raise RuntimeError('Pressure test needs a dedicated child cgroup; container root is insufficient')
    group = cgroup_root / relative.lstrip('/')
    limit = (group / 'memory.max').read_text().strip()
    if limit == 'max' or not 0 < int(limit) <= 32 * 2**30:
        raise RuntimeError('Dedicated cgroup memory.max must be positive and at most 32 GiB')
    if (group / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('Dedicated cgroup memory.swap.max must be 0')
    if (group / 'memory.oom.group').read_text().strip() != '1':
        raise RuntimeError('Dedicated cgroup memory.oom.group must be 1 to kill all test workers together')
    processes = (group / 'cgroup.procs').read_text().split()
    if processes != [str(os.getpid())]:
        raise RuntimeError('Dedicated cgroup must initially contain only this test process')
    return group, int(limit)


def memory_snapshot(group):
    result = {}
    for name in ('memory.current', 'memory.peak', 'memory.events', 'memory.stat'):
        path = group / name
        if path.exists():
            result[name] = path.read_text().strip()
    usage = os.statvfs('/dev/shm')
    result['shm_used_bytes'] = (usage.f_blocks - usage.f_bfree) * usage.f_frsize
    return result


def record(path, **values):
    """Flush each progress record so an abrupt cgroup kill retains previous samples."""
    with Path(path).open('a') as handle:
        handle.write(json.dumps(values, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
