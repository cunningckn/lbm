# Validation and reproducible entry points

The September 2026 cleanup was tested on js_dev_2 (A800 80 GB GPUs) in
isolated checkouts. Dataset files and the user's checkout were preserved.

| Work | Evidence |
| --- | --- |
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
statistics are not production defaults. Exact quantiles still retain the
state/action observations in memory: for extremely large datasets this
separate statistics computation needs capacity planning. The bounded-memory
mmap packing result does not imply bounded exact-quantile computation.

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
are included. Future precomputed vision-feature support should have an
explicit batch contract and separate benchmark label; it is not currently
implemented or included in the throughput claims.

Hosted CPU CI checks regressions without private datasets, CUDA or downloaded
weights. Server integration is additional coverage. The 200+5-step real run
checks numerical stability and restore, not task convergence, held-out
accuracy or multi-day training stability. Closed-loop evaluation still
requires a trained checkpoint and a chosen LIBERO/RMBench task protocol.
