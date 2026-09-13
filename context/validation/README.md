# Validation and reproducible entry points

The September 2026 cleanup was tested on js_dev_2 (A800 80 GB GPUs) in
isolated checkouts. Existing source datasets and uncommitted checkout edits
were preserved.

The earlier runtime regression on the PR #24 code passed **463 tests, zero skips**
with GPUs, pretrained assets and real datasets enabled. The new hosted CPU
job in PR #25 passed **432 tests**, with **17 explicit resource-dependent
skips and 14 integration cases deselected**. CPU CI does not replace the
server run. CLI help smoke checks passed for scan, mmap, norm, FK,
inspection, training and the training-throughput benchmark.

The earlier follow-up added bounded exact normalization, frozen vision caches,
compact prefix conditioning and compact image transfer. That two-GPU suite
passed 493 tests with zero skips, including mixed-checkpoint inference.
Its CPU regression with pretrained assets passed 468 with 11 resource skips
and 14 integration deselections. The subsequent audit passed 507 full-suite
cases plus a newly added killed-writer recovery case; see the audit below
for completed checks and the server-connectivity blocker.

Latest follow-up: [matched workload tests and conditioning optimization](matched/README.md).
Previous: [validation, recovery and feature-shard audit](audit/README.md).

| Work | Evidence |
| --- | --- |
| Bounded exact normalization memory | [normalization](bounded-norm/README.md) |
| Frozen vision feature preparation | [feature cache](feature-cache/README.md) |
| Long held-out training and LIBERO closed loop | [quality validation](remaining/README.md) |
| Comparable real-data throughput tuning | [real throughput](real-throughput/README.md) |
| GPU + pretrained + real-data initial baseline | [initial validation](2026-09-13/README.md) |
| Single/dual GPU throughput and multi-source loader tuning | [scaling and mixture](scaling-mixture/README.md) |
| Bounded mmap packing, corruption and interruption recovery | [cache integrity](cache-integrity/README.md) |
| Real normalized training and checkpoint restore | [stability](normalized-stability/README.md) |

Single-A800 resident synthetic complete updates reached 263.65 samples/s at
batch 384; dual-GPU FSDP reached 489.09 samples/s at batch 256 per rank.
Forward/backward-only measurement reached 270.03 samples/s at batch 256.
These runs still compute frozen DINO online. They do not measure a
precomputed-image-feature path. Real Kai0+Agibot batches reached about
70 samples/s with eight workers in the short loader trial. Image windows,
action length, precision, update scope and data source must match when
comparing throughput. See the reports for complete settings and limitations.

After the exact-video-seek fix and training-loop refactor, the real mixture
trial was repeated for 40 updates at batch 32, eight workers, no mmap and
the same one-episode-per-source selection. Steps 20/30/40 logged
2.20/2.21/2.18 updates/s (70.40/70.72/69.76 samples/s), with 33,249.52 MiB
peak allocated memory. This matches the earlier throughput baseline while
using correctly decoded target frames. Norm files were absent in this
throughput-only trial; normalized stability is a separate experiment below.

## Prepare and train multiple datasets

Use Python 3.12 and the locked GPU environment from the README. Place
prepared data in `datasets/<registered_name>` and pretrained encoders in
`checkpoints/` (or configure their documented path overrides).

```sh
uv run python scripts/build_scan_index.py --dataset kai0,agibot
uv run python scripts/prebuild_mmap.py --dataset kai0,agibot --workers 8
uv run python scripts/compute_norm.py --dataset kai0,agibot --action-mode delta
uv run python scripts/train.py --data-mix kai0,agibot --action-mode delta \
  --batch-size 32 --num-workers 8 --output-dir outputs/mixed
```

Use explicit action frequency/length/kind/format consistently across norm
generation and training when overriding defaults. Production statistics
should cover representative training data; the one-episode validation
statistics are not production defaults. Exact quantiles spool float32 observations to temporary disk and select ranks
in fixed-size blocks. Use `compute_norm.py --stats-temp-dir /scratch` to choose
a disk with enough space for the state/action rows. Dataset adapters still
load episode vectors; the reducer no longer retains the whole corpus.

For reproducible standard-checkpoint replay, start with zero workers and
resume with the same model/data settings; `--resume` restores optimizer,
scheduler and replay state, while `--ckpt` initializes policy weights.

```sh
uv run python scripts/train.py --data-mix kai0,agibot --num-workers 0 \
  --output-dir outputs/replay --steps 200 --ckpt-every 200
uv run python scripts/train.py --data-mix kai0,agibot --num-workers 0 \
  --output-dir outputs/replay --resume outputs/replay/last.pt --steps 205
```

## Measure and regress

```sh
PYTHONPATH=src python benchmarks/train_throughput.py \
  --batch-sizes 32,64,128,256 --image-size 224 --warmup 5 --iters 20 --json
PYTHONPATH=src python benchmarks/train_throughput.py \
  --batch-sizes 256 --warmup 5 --iters 20 --no-optimizer-step --json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LBM_REAL_DATA=1 \
  uv run --locked --extra dev --extra data pytest -q --require-pretrained-assets
```

Increase batch size sequentially on an idle GPU; an OOM is a capacity limit,
not a successful measurement. The benchmark separates reused synthetic
batches from newly generated inputs and records whether optimizer updates
are included. Frozen vision-feature caches now have an explicit batch contract
and a separate real-data benchmark label; they are not included in the older
synthetic numbers above.

Hosted CPU CI checks regressions without private datasets, CUDA or downloaded
weights. Server integration is additional coverage. The follow-up completed
5,000+10 real mixed-data updates with episode-disjoint validation and a
separate LIBERO task's trained-policy closed-loop evaluation (8/10 successes).
These bounded experiments do not establish full-suite accuracy, convergence
or multi-day training stability.
