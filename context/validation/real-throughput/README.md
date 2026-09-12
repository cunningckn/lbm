# Comparable real-data throughput tuning

Hardware: js_dev_2, NVIDIA A800-SXM4-80GB, Python 3.12.13, PyTorch
2.7.1+cu126. Each single-GPU experiment uses one GPU, OMP/MKL threads 2.

The workload is the same eight-episode Kai0+Agibot subset (11,829 samples),
training-only normalization, full default 2,015.9M-parameter BF16 policy,
frozen DINO, three camera slots, 224-pixel images, one history frame,
150 action steps, maximum prefix 4, and CLIP task vectors. The features
were generated from actual source observations, not random tensors.

`run.py` measures 60 complete training updates after 10 warmups. The timing
includes DataLoader wait, device transfer, forward, backward, gradient
clipping, AdamW and loop overhead. A CUDA synchronization at each optimizer
completion makes the timing explicit. Validation and checkpoint saving do
not occur during these throughput runs. Samples/s means samples processed,
not necessarily distinct samples. Increasing batch size changes optimization
behavior; throughput alone does not prescribe a learning-rate schedule.

## Single GPU

| Path | Batch | Workers | Samples/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Features, original dense prefix projections | 96 | 2 | 98.75 | 68.34 |
| Features, compact prefix projections | 96 | 2 | 117.54 | 58.74 |
| Features, compact prefix projections | 128 | 0 | 90.46 | 73.55 |
| Features, compact prefix projections | 128 | 2 | 123.35 | 73.55 |
| Features, compact prefix projections | 128 | 4 | 123.29 | 73.55 |
| Features, compact prefix projections | 136 | 2 | 124.58 | 77.39 |
| Live images, CPU normalization | 32 | 8 | 72.65 | 29.28 |
| Live images, GPU normalization | 32 | 8 | 79.36 | 29.28 |
| Live images, GPU normalization | 64 | 4 | 95.03 | 44.00 |
| Live images, GPU normalization | 64 | 8 | 95.03 | 44.00 |
| Live images, GPU normalization | 64 | 16 | 95.10 | 44.00 |
| Live images, GPU normalization | 96 | 8 | 100.85 | 58.65 |
| Live images, GPU normalization | 128 | 4 | 97.54 | 73.44 |
| Live images, GPU normalization | 128 | 8 | 105.76 | 73.44 |
| JPEG mmap, GPU normalization | 96 | 4 | 100.99 | 58.65 |
| JPEG mmap, GPU normalization | 128 | 4 | 105.76 | 73.44 |
| JPEG mmap, GPU normalization | 128 | 8 | 105.67 | 73.44 |

Compact prefix projection improves the matched batch-96 cached workload by
19.0%, reducing peak allocated memory by 9.59 GiB. Moving uint8 frames before
normalizing improves the matched batch-32 live workload by 9.2%. Existing
JPEG mmap may re-encode pixels; it is a separate input path, not a numerical
identity claim about the source images.

The largest successful cached batch tested is 136, at 124.58 samples/s, reproduced at 124.76 samples/s.
Batch 144/160/192 exceeded available memory. An independent batch-128/two-worker repeat measured 123.32 samples/s,
consistent with 123.35 in the first run. Batch 128 delivers about 99% of
that throughput with nearly 4 GiB less allocated memory and is the practical
single-GPU recommendation. Two workers hide cache IO; zero workers are
slower, and four do not improve throughput here. For online vision use
batch 128/workers 8, or mmap/batch 128/workers 4. Their throughput is equal
within measurement noise; mmap reduces the workers needed in this test.

Feature caching is best among these measured single-GPU configurations when
the vision backbone is frozen. The cache is about 21 GB for 11,829 samples;
build time and disk capacity must be amortized over training. It cannot
replace live images when training the vision backbone or changing image
augmentation/preprocessing. These subset/host measurements do not establish
a universal optimum for different horizons, encoders, storage or GPUs.

The earlier synthetic 263.65 samples/s baseline had only 50 action steps
and no prefix conditioning. It is not an apples-to-apples denominator for
these real 150-step measurements, or proof that real IO loses that ratio.

## Two GPUs

The same real feature-cache workload with Megatron-FSDP, BF16 compute and
its default FP32 master parameters/gradients:

| Per-rank batch | Workers/rank | Block group | Global samples/s | Max-rank allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| 96 | 2 | 1 | 217.61 | 63.15 |
| 112 | 2 | 1 | 224.25 | 71.03 |
| 120 | 2 | 1 | 229.22 | 74.17 |
| 112 | 2 | 2 | 224.17 | 71.50 |

Global throughput is `world * batch * measured_updates / max(rank_seconds)`;
both rank measurements are retained. Batch 128 per rank exceeded memory.
An independent repeat measured 228.49 global samples/s, within 0.4% of
the first run. The best measured dual-GPU candidate is batch 120 per rank, two workers per
rank, default group 1. Group 2 did not help. This is weak scaling with a
larger global batch, and FSDP's master-state precision differs from the
standard BF16 optimizer; it is not an exact optimizer-numerics comparison.

Feature caches were on local `/tmp` scratch storage; original sources and
JPEG mmap files were on the dataset filesystem. Put production feature
caches on sufficiently large local scratch storage to reproduce this IO
regime. A cache on slower shared storage needs its own measurement.

## Reproduce

Run from repository root with the prepared datasets and local encoder
weights. The experiment uses the training-only norm files created by the
mixed-data stability harness; pass their directory explicitly.

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  python context/validation/real-throughput/run.py --mode features \
  --cache /scratch/features --batch 128 --workers 2 \
  --data-root /data/prepared --norm-dir /scratch/long-heldout \
  --output /scratch/benchmark-features
# Matched prefix ablation: add --dense.
# Online path: --mode live; add --mmap for prepared mmap caches.
```

`--baseline-batch-module` accepts the original `batch.py` for the CPU-image
normalization ablation; use the file from commit `5a4786d`.

For the measured dual-GPU configuration:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 context/validation/real-throughput/run.py \
  --fsdp --mode features --cache /scratch/features --batch 120 --workers 2 \
  --data-root /data/prepared --norm-dir /scratch/long-heldout \
  --output /scratch/benchmark-fsdp
```

Per-rank timings and memory measurements: [aggregate results](results.json).
