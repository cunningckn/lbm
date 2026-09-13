"""Out-of-RAM IO stress using physical copies of real features, not new training examples."""
import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

import torch
from context.validation.audit.pressure_guard import check_isolation, memory_snapshot, record

from lbm.config import TrainConfig
from lbm.feature_shards import atomic_json, file_sha256
from lbm.training_features import FeatureDataset
from lbm.training_loader import make_loader

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--report', required=True)
parser.add_argument('--copies', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--workers', type=int, choices=[0, 1, 2], default=1)
args = parser.parse_args()
# Check before any copying or worker creation. Children inherit this hard limit.
group, memory_limit = check_isolation()
report_path = Path(args.report).resolve()
if report_path.is_relative_to('/tmp') or report_path.is_relative_to('/dev/shm'):
    raise ValueError('report must be on persistent storage outside /tmp and /dev/shm')
report_path.parent.mkdir(parents=True, exist_ok=True)
progress_path = report_path.with_suffix('.progress.jsonl')
record(progress_path, phase='start', cgroup=str(group), memory_limit_bytes=memory_limit,
       memory=memory_snapshot(group))
if args.copies < 1 or not 1 <= args.batch_size <= 128:
    raise ValueError('copies must be positive and batch-size must be in [1, 128]')
root, target = Path(args.source), Path(args.output)
if target.exists():
    raise FileExistsError(target)
base = json.loads((root / 'manifest.json').read_text())
size = sum(p.stat().st_size for p in root.rglob('*.npy')) * args.copies
physical_ram = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith('MemTotal:'))) * 1024
if size <= memory_limit:
    raise ValueError('increase copies: physical cache must exceed dedicated cgroup memory.max')
if shutil.disk_usage(target.parent).free < size + 20 * 2**30:
    raise ValueError('insufficient scratch space')
target.mkdir()
try:
    metadata = dict(base['metadata'], sources=[], stress_replication=args.copies)
    shards = []
    start = time.perf_counter()
    for _ in range(args.copies):
        for entry in base['shards']:
            src = root / entry['directory']
            dest = target / f'{len(shards):06d}'
            dest.mkdir()
            for table in src.glob('*.npy'):
                shutil.copyfile(table, dest / table.name)
            manifest = json.loads((src / 'manifest.json').read_text())
            manifest['metadata'] = metadata
            atomic_json(dest / 'manifest.json', manifest)
            shards.append(dict(directory=dest.name, rows=entry['rows'],
                               sha256=file_sha256(dest / 'manifest.json')))
        print(f'COPIED {len(shards)} shards', flush=True)
        record(progress_path, phase='copy', shards=len(shards), memory=memory_snapshot(group))
    atomic_json(target / 'manifest.json', dict(version=2, rows=base['rows'] * args.copies,
                                              metadata=metadata, shards=shards))
    build_seconds = time.perf_counter() - start
    dataset = FeatureDataset(target)
    cfg = TrainConfig(batch_size=args.batch_size, num_workers=args.workers)
    cfg.data.prefetch_factor = 1
    cfg.data.pin_memory = False
    loader, _ = make_loader(dataset, config=cfg, distributed=False, train=True,
                            collate_fn=torch.utils.data.default_collate)
    ends, workers_rss = [], []
    for i, batch in enumerate(loader):
        assert torch.isfinite(batch['state']).all()
        ends.append(time.perf_counter())
        child_pids = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
        rss = 0
        for pid in child_pids:
            try:
                rss += int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            except FileNotFoundError:
                pass
        workers_rss.append(rss)
        record(progress_path, phase='read', batch=i, memory=memory_snapshot(group))
        if i == 109:
            break
    elapsed = ends[-1] - ends[9]
    report = dict(physical_bytes=size, host_ram_bytes=physical_ram, memory_limit_bytes=memory_limit, rows=len(dataset),
                  duplicated_real_features=True, build_seconds=build_seconds,
                  batch_size=cfg.batch_size, workers=cfg.num_workers,
                  measured_batches=100, samples_per_second=100*cfg.batch_size/elapsed,
                  parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  max_worker_rss_bytes=max(workers_rss))
    atomic_json(report_path, report)
    record(progress_path, phase='complete', memory=memory_snapshot(group))
    print('PRESSURE ' + json.dumps(report), flush=True)
    del batch, loader, dataset
finally:
    shutil.rmtree(target)
    record(progress_path, phase='cleaned')
