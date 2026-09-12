# A800 scaling and mixture baseline

Environment: js_dev_2, NVIDIA A800-SXM4-80GB (two idle GPUs), PyTorch
2.7.1+cu126, Python 3.12.13, `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`.
Use `/mnt/kpfs/workspace/jinaoqun/Projects/lbm/.venv/bin/` executables.
The synthetic scaling script is unchanged from `df8acd9`; backward-only mode
uses MR #17 (`5bc895c`). No encoder features were cached.

## Synthetic complete updates

Default DiTConfig, BF16, frozen DINO with online forward, language=none,
224px images, repeated resident synthetic batch. Includes forward, backward
and AdamW update. Five warmup iterations, 30 measured for batch <=256 in
single-GPU runs and 20 for larger/double-GPU runs. Separate CUDA-event
profiles use three iterations and need not exactly sum to wall time.

| GPU count | Mode | Per-GPU batch | Global samples/s | Rank-0 peak MiB |
| ---: | --- | ---: | ---: | ---: |
| 1 | dense | 16 | 105.46 | 18655.5 |
| 1 | dense | 32 | 156.41 | 18695.7 |
| 1 | dense | 64 | 198.81 | 20316.1 |
| 1 | dense | 128 | 235.37 | 29281.9 |
| 1 | dense | 256 | 255.31 | 47344.0 |
| 1 | dense | 384 | 263.65 | 65124.8 |
| 1 | dense | 512 | OOM | — |
| 2 | FSDP | 128 | 432.84 | 33791.6 |
| 2 | FSDP | 256 | 489.09 | 51873.4 |

At per-rank batch 256, weak-scaling efficiency is about 95.8% relative to
single-GPU throughput (489.09 / (2 * 255.31)). Global throughput is the existing
benchmark's rank-0 estimate times world size, not independently reduced
slowest-rank timing. This is weak scaling, not a fixed-global-batch comparison.
Batch 384 offers only ~3.3% more throughput than 256 while using much more
memory. 256 is a practical baseline here, not a universal production default.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/throughput_fsdp.py --no-fsdp --batch-sizes 16,32,64 --warmup 5 --iters 30 --profile --no-util
CUDA_VISIBLE_DEVICES=0 python benchmarks/throughput_fsdp.py --no-fsdp --batch-sizes 128,256 --warmup 5 --iters 30 --profile --no-util
CUDA_VISIBLE_DEVICES=0 python benchmarks/throughput_fsdp.py --no-fsdp --batch-sizes 384,512 --warmup 5 --iters 20 --profile --no-util
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 benchmarks/throughput_fsdp.py --batch-sizes 128,256 --warmup 5 --iters 20 --profile --no-util
```

Raw logs: [small batches](dense-small.txt), [large batches](dense-large.txt),
[capacity](dense-capacity.txt), [FSDP](fsdp.txt).

## Forward/backward only

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/train_throughput.py --device cuda --batch-sizes 256 --warmup 5 --iters 20 --no-optimizer-step --json
```

Resident-input compute: **270.03 samples/s**. Including fresh synthetic input
preparation: **59.87 samples/s**; input preparation averaged 3443.16 ms/step.
These are distinct measurements, not real dataset performance. Peak allocated
memory across both phases: 40198.37 MiB. Online frozen DINO remains included;
this does not measure a precomputed-feature or backbone-bypassed DiT.

## Real Kai0 + Agibot mixture

[Reproduction script](mix_bench.py): one episode per dataset (3095 samples
combined), default pretrained DINO, BF16, batch 32, 40 updates, log every 10.
Model uses state dimension 20, action dimension 22 and three cameras;
action horizon becomes 150 steps (5s at 30 Hz). No feature precomputation.
Run from the isolated checkout with `CUDA_VISIBLE_DEVICES=0`, using
`--workers 2`, `--workers 8`, then `--workers 8 --mmap`.

| Workers | mmap | Last three interval step/s | Approx samples/s |
| ---: | --- | --- | ---: |
| 2 | off | 1.51, 1.52, 1.54 | 48–49 |
| 8 | off | 2.21, 2.20, 2.20 | 70–71 |
| 8 | on | 2.20, 2.18, 2.20 | ~70 |

Peak allocated memory: 33249.52 MiB in all three runs. Loss/gradient logs
remain finite; small numerical differences with mmap are visible in the logs.
The first interval includes startup and is excluded from the ranges above.
Existing OS/cache warming and short windows limit the comparison; do not
interpret this as proof mmap is useless on larger, colder datasets. The worker
comparison suggests retaining the existing eight-worker default here; there
is no evidence to change global prefetch or mmap defaults.

Raw logs: [two workers](mix-w2.txt), [eight workers](mix-w8.txt),
[mmap](mix-mmap.txt). Norm stats are absent for both datasets: this verifies
loading/training stability, not normalized convergence or final quality.
