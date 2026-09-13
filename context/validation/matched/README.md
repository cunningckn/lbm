# Matched training throughput and conditioning audit — 2026-09-13

This review starts from `d07e974` (merged PR #39). Implementation commit
`75fb031` is tested here. PR #40 completes the current detailed normalization
review; review again after five subsequent merged PRs. See `CONTRIBUTING.md` for the checklist.

## Workload and method

Measurements use js_dev_2, one A800-SXM4-80GB, PyTorch 2.7.1+cu126, Python
3.12.13, BF16, and the full 2,015,935,510-parameter model (1,928,854,294 trainable).
The original server checkout and its uncommitted preprocessing script are
preserved; execution uses `/tmp/lbm-audit` with `PYTHONPATH=src:.`.

The Kai0+Agibot reference has 8,734 training and 3,095 held-out rows, from four
episodes per source before an episode split (seed 123, holdout 0.25). Statistics
use only training episodes. Three 224px cameras, one image per camera, state
width 20, action width 22 and **150 action steps** are identical across modes.
Frozen DINO is either evaluated online or represented by its cached features.
The default prefix setting 4 samples lengths **0–3** (exclusive upper bound),
with probability 1, noise scale 0.05 and state masking probability 0.1.

All main measurements use 10 warmup updates, 60 measured updates, then 10
separate synchronized diagnostic updates. Updates include forward, backward,
gradient clipping, AdamW and the learning-rate scheduler. Finite losses and
gradient norms are checked. Compilation is included in warmup, excluded from
steady throughput; dataset/model setup precedes warmup. This benchmark does
not include validation or checkpoint I/O in throughput.

The six baseline modes share contract SHA
`f8f88ad0f80e32046fff4c34a6efe9d82ee1979d1c2fd961a6548183d5d90fcb` and cache
manifest SHA `f203e8e7bc9b01e116218918e4fb2ada50a19f3c17a4e698657c0b4527445a2d`.
Online modes verify source ordering, episode identity and normalization against
the cache fingerprint. Changing batch to 160 changes the contract hash; those
rows are a batch-capacity comparison, not the same-batch comparison.

## Six matching baselines

| Input mode | Batch | Workers | sample/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Synthetic GPU features | 128 | — | 124.25 | 73.33 |
| Reused real GPU features | 128 | — | 123.99 | 73.33 |
| Shuffled real feature cache | 128 | 2 | 121.84 | 73.33 |
| Synthetic GPU images + DINO | 128 | — | 105.71 | 73.44 |
| Reused real GPU images + DINO | 128 | — | 105.56 | 73.44 |
| Shuffled real online images + DINO | 128 | 8 | 97.00 | 73.45 |

Resident modes reuse one batch. Synthetic floating tensors are random and
masks are all valid; they measure a compute ceiling, not learning quality.
The historical 263.65 sample/s run used 50 actions and no prefix conditioning;
it must not be compared directly with this 150-action training recipe.

The cached baseline diagnostic spends about 309 ms in forward, 657 ms in
backward/zero-grad, 57 ms in AdamW, and 10 ms preparing/transferring the batch.
Online DINO adds about 179 ms. The operator diagnostic identifies expanded
prefix `where` work as a material cost (~12.5% self CUDA time). Operator and
kernel rows in a profiler table overlap and must not be summed together.

## Optimization decisions

`models/conditioning.py` now owns prefix projection, modulation and gating.
Only the possible prefix window is expanded; the remaining timesteps share a
broadcast condition. `Tensor.split` avoids the full-sized zero buffers created
by separate slice backward operations. No parameters or checkpoint keys change.

| Cached experiment | sample/s | Peak allocated GiB | Decision |
| --- | ---: | ---: | --- |
| Baseline | 121.84 | 73.33 | Control |
| Initial separate slices | 119.10 | 63.04 | Rejected: slower backward |
| Native split-prefix | 126.94 | 63.04 | Enabled in production |
| Fused AdamW alone | 125.69 | 73.33 | Explicit option |
| Split + compiled pointwise, original casting | 144.81 | 63.00 | Preliminary; replaced by precision-preserving compilation |
| Split + compiled pointwise + fused AdamW | 148.38 / 149.15 | 63.00 | Same-GPU original/repeat |
| Same combination, batch 160 | 151.74 | 74.95 | Highest cache throughput tested; little memory headroom |

The final compiler configuration preserves BF16 intermediate casts. Backward
reductions and fused AdamW still change floating-point rounding; neither is a
promise of bitwise equality to the old backend. These remain opt-in with
`--compile-conditioning --fused-adamw`. Full-model `--compile` is mutually
exclusive with conditioning compilation. Optimizer settings and parameter
groups come from the shared `build_adamw` factory.

The final combination reduces cached diagnostic forward to ~250 ms, backward
to ~558–560 ms and AdamW to ~22 ms. Batch-128 warmup took 29–33 seconds in the
recorded runs, including compilation/cache reuse; this is not a portable cold
compile-time estimate. New shapes can trigger more compilation.

At the original eight workers/prefetch 2, online throughput improves to 113.98
sample/s at batch 128 and 115.55 at batch 160. A same-GPU baseline repeat gives
97.14, confirming the original 97.00. The normal-pass wait distribution exposes
an epoch-boundary stall (~5.7–7.4 seconds); p95 loader wait remains under 1 ms.
Synchronized diagnostic stages alone hide this tail because they run after the
boundary. The aggregate JSON therefore also records normal-pass mean/p95/max
loader and step times.

The original online recipe disables both vector and image mmap and decodes
videos directly. Worker/prefetch tuning of this path gave:

| Workers | Prefetch per worker | sample/s | Mean loader wait ms |
| ---: | ---: | ---: | ---: |
| 2 | 1 | 51.80 | 1443.1 |
| 4 | 1 | 89.62 | 406.5 |
| 8 | 1 | 113.53 | 102.4 |
| 8 | 2 | 113.98 | 95.9 |

Reducing workers does not help this video decoder workload. Keep eight workers
and prefetch 2 here; prefetch 1 offers no convincing speed gain. The eight-worker
prefetch-1 run overlapped a short read-only cache readiness check near its end;
its sub-percent difference is not evidence of a meaningful ranking.

`--mmap-images` is a separate production-path comparison: existing JPEG mmap
frames are decoded, then DINO still runs online. It requires online image mode
and disables cache construction during measurement. `prepare_mmap.py` checks
or prepares only the reference training episodes with two workers; all 18
camera streams were already cached in this run (zero builds, 18 reuses).
JPEG quality is 85. Episode identity and normalization match the reference,
but JPEG pixels need not be bitwise identical to raw video inputs. The shared
model/optimizer contract hash alone does not imply identical image encoding.

| JPEG mmap experiment | Batch | Workers | sample/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 128 | 4 | 102.96 | 73.44 |
| Split + compiled conditioning + fused AdamW | 128 | 4 | 121.33 | 63.11 |
| Same combination | 128 | 2 | 121.85 | 63.11 |
| Same combination | 160 | 4 | 123.37 | 75.09 |
| Same combination | 160 | 2 | 123.63 | 75.09 |

For this workload, use the combination with **batch 128, two workers and
prefetch 2** for cached features or JPEG mmap images; use eight workers for
raw video. Two/four mmap workers are effectively tied in throughput, so two
uses fewer CPU resources. Batch 128 already captures most of the speedup and
leaves about 12 GiB more allocation headroom than batch 160. The highest tested
throughput is 151.74 sample/s with features and 123.63 with mmap at batch 160.
These are the best measured points in this bounded comparison, not a global
optimum over all datasets, batch sizes or training recipes. No default worker
or prefetch value was changed based on this small subset.

## Reproduction

Use the feature-cache preparation process in the previous audit to obtain
matching train/validation caches. The server artifacts for this run are under
`/mnt/kpfs/workspace/jinaoqun/Projects/lbm-validation-20260913/matched`.
Run `PYTHONPATH=src:. python context/validation/matched/prepare_mmap.py
--cache /path/features-train --data-root /path/datasets` before the mmap trial.
JSON files committed here contain only aggregate metrics and configurations.
No dataset arrays, normalization values or checkpoints are published.

```bash
export PYTHONPATH=src:. OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
python benchmarks/matched_throughput.py \
  --mode features --cache /path/features-train --data-root /path/datasets \
  --batch 128 --workers 2 --output /path/baseline.json
python benchmarks/matched_throughput.py \
  --mode features --cache /path/features-train --data-root /path/datasets \
  --batch 128 --workers 2 --split-prefix --compile-conditioning --fused-adamw \
  --output /path/optimized.json
# Raw online: --mode live --workers 8; prefetch comes from cache provenance.
# Existing JPEG mmap online: additionally pass --mmap-images.
# --prefetch-factor overrides the loader's batches per worker for tuning.
```

The benchmark defaults to the old expanded prefix representation as a control.
Production uses split-prefix by default. Set `--split-prefix` when measuring
production behavior. Each result path must be new; the tool refuses overwrite.

A production CLI check used real train and validation caches, batch 32, two
workers, 40 updates, validation every 20 updates (two batches), and checkpoint
at 40. A fresh process resumed `last.pt` to total step 44 with the same options:

```bash
python scripts/train.py --feature-cache /path/features-train \
  --val-dataset /path/features-val --output-dir /path/run \
  --compile-conditioning --fused-adamw --batch-size 32 --num-workers 2 \
  --steps 40 --val-every 20 --val-batches 2 --ckpt-every 40 \
  --log-every 10 --no-dump-batch
# Repeat in a fresh process with --resume /path/run/last.pt --steps 44.
```

The complete 2B-parameter two-GPU FSDP CLI also passed: two updates, held-out
validation and sharded save, then a fresh `torchrun` invocation loading step 2,
continuing to step 4, validating and saving again. This used batch 8 per rank,
two loader workers, both acceleration options and the same real caches. To
reproduce, prefix the CLI above with `torchrun --standalone --nproc_per_node=2`,
add `--fsdp`, set batch 8, steps/validation/checkpoint interval 2 and log interval
1; then resume the directory `run/2` with total steps 4. The reconstruction
metrics were finite (9.0244 at step 2, 8.9879 at step 4); these very short runs
establish execution/recovery, not policy quality. Both GPUs were idle afterward
and the container's `oom`/`oom_kill` event counters remained zero.

Checkpoint resume signatures include enabled acceleration options. Disabled
new options are omitted to retain compatibility with legacy default signatures.
Resuming requires the same enabled options. Mathematical compatibility across
code versions does not guarantee identical floating-point training trajectories.

## Correctness and normalization review

- Native split loss/gradient checks cover FP32 and BF16, probability 0/0.5/1,
  no prefix, partial prefix, whole-chunk prefix, padded actions and masked state.
- Compiled FP32 gradients use strict tolerances. Compiled BF16 checks bound each
  gradient tensor's relative L2 error by two BF16 epsilons, with a 1e-8 floor for
  very small gradients. Measured worst relative L2 errors were 1.49% without
  prefix and 1.32% with prefix; forward loss matched in these fixed tiny cases.
  This is a rounding-error check, not evidence of equal long-run convergence.
- Fused AdamW tests bound update rounding for FP32/BF16 and require bitwise
  identical parameters and optimizer state after interrupted/resumed updates.
- Single-GPU regression with real data and required pretrained assets passed
  529 tests. All 19 distributed tests also passed with two GPUs, including
  accelerated cached FSDP exact loss replay after interruption. Follow-up
  prefetch/CLI/loader tests passed 34 tests and mmap-mode tests passed 12
  (overlapping, not additive to the full-suite count).
- The actual legacy `profile_split.py --batch-size 4 --warmup 1 --iters 1
  --kernels` GPU run completed. Its obsolete wrappers now forward keyword
  arguments and restore methods even on failure; a regression test covers this.
- Documentation distinguishes synthetic inference latency from a LIBERO closed
  loop. Dataset registry, shared loader/optimizer/defaults and checkpoint
  compatibility were reviewed; no new per-dataset branches were introduced.

This is a short throughput/correctness comparison on a bounded mixed-data
sample, not a multi-day convergence study or a full LIBERO-suite evaluation.
The existing capacity-pressure check remains blocked by the server's read-only
cgroup hierarchy. No unbounded RAM/SHM exhaustion experiment was attempted.
