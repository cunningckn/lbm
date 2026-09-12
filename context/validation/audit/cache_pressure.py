"""Out-of-RAM IO stress using physical copies of real features, not new training examples."""
import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

import torch

from lbm.config import TrainConfig
from lbm.feature_shards import atomic_json, file_sha256
from lbm.training_features import FeatureDataset
from lbm.training_loader import make_loader

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--report', required=True)
parser.add_argument('--copies', type=int, default=16)
args = parser.parse_args()
root, target = Path(args.source), Path(args.output)
if target.exists():
    raise FileExistsError(target)
base = json.loads((root / 'manifest.json').read_text())
size = sum(p.stat().st_size for p in root.rglob('*.npy')) * args.copies
physical_ram = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith('MemTotal:'))) * 1024
if size <= physical_ram:
    raise ValueError('increase copies: physical cache must exceed host RAM')
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
    atomic_json(target / 'manifest.json', dict(version=2, rows=base['rows'] * args.copies,
                                              metadata=metadata, shards=shards))
    build_seconds = time.perf_counter() - start
    dataset = FeatureDataset(target)
    cfg = TrainConfig(batch_size=128, num_workers=2)
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
        if i == 109:
            break
    elapsed = ends[-1] - ends[9]
    report = dict(physical_bytes=size, host_ram_bytes=physical_ram, rows=len(dataset),
                  duplicated_real_features=True, build_seconds=build_seconds,
                  measured_batches=100, samples_per_second=100*128/elapsed,
                  parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  max_worker_rss_bytes=max(workers_rss))
    Path(args.report).write_text(json.dumps(report, indent=2))
    print('PRESSURE ' + json.dumps(report), flush=True)
    del batch, loader, dataset
finally:
    shutil.rmtree(target)
