# Dataset-local frozen features — 2026-09-14

The prebuild entrypoint defaults to
`<dataset>/.cache/<encoder>/prebuilt/<contract SHA256>/<split>` for one source.
The shell wrapper is `scripts/prebuild_visual.sh`. Explicit output paths still
support mixtures and existing workflows. Completed caches require explicit
`--reuse-cache`, which validates metadata and every payload checksum. Interrupted
builds use the existing locked, resumable shard writer. No full preprocessing
was launched during these checks.

Identity includes source/episode/normalization contracts, actual encoder and
text-model weights, tokenizer, input/history configuration, image dimensions,
precision/device type and preprocessing code. Train/validation caches also reject
incompatible extraction contracts. Prepared history ages remain FP32 when other
floating policy inputs are converted to BF16, matching the live path.

## Real-data screening

A800 80GB, full 2,022.8M-parameter policy, BF16, batch 64, two workers, one-second
30-position actions, three observations at 10Hz for both visual/state history.
Two actual Agibot episodes in subtask mode yield 2,482 samples. Prebuild batch32,
128-row shards, three cameras. The cache contains 12.60 GiB of array payloads.
No source normalization was available in this screening, so both paths use raw
state/action values; these runs do not measure task quality. All paths run 100
updates, crossing multiple epochs, with finite losses and gradient norms.

| Path | Samples/s (updates 6–100) | Peak allocated GPU GiB |
|---|---:|---:|
| Online frozen DINO | 112.00 | 22.39 |
| Shared-filesystem cache | 22.58 | 22.56 |
| Local-disk copy of same cache | 215.85 | 22.56 |

Rates are total measured samples divided by summed five-update logger intervals;
first five updates are excluded. Setup, extraction, copy and validation time are
excluded. The shared-storage run completed, including slow first shard accesses;
its poor result must not be hidden by reporting only the final fast intervals.
The local-copy path is 1.93x the online rate in this single screening, used only
to diagnose the shared-storage bottleneck. The user requires cloud storage in
production; local staging is not the proposed solution. No completed cache was
deleted or overwritten.

The three above runs precede the FP32-age casting correction. They establish a
storage bottleneck and a candidate deployment layout, not final corrected-code
numerical equivalence. Initial shared weights, state/action input hashes and CUDA
RNG match across the first three batches. CLIP vector hashes differ: a separate
same-sample comparison measured maximum absolute error 1.91e-6, relative L2 below
8.67e-7 and cosine similarity at least 0.9999998. CPU text batching introduces tiny
rounding differences; optimization trajectories are not bitwise-identical.
Shared/local cached runs have identical logged losses. Do not equate this with
identical live/cached training trajectories or proven equal policy quality.

Evidence: `live100.json`, `shared100.json`, `local100.json`, `text-parity.json`.
Raw features and checkpoints stay on the server. `run_matched.py` reproduces the
bounded online/cache workload; `check_text.py` compares text outputs by sample.


## Corrected-code direct cloud mmap checks

The final code completed two further 100-update runs directly against the cloud
cache, retaining up to 32 shard mappings per process (20 shards in this fixture).
The payload remains mmap-backed; there is no local-copy training or whole-cache
RAM preload in these runs. All loss logs and input/RNG hashes agree between the
two worker configurations.

| Direct cloud mmap | Samples/s, updates 6–100 | Peak allocated GPU GiB |
|---|---:|---:|
| 32 open shards, 2 workers | 217.24 | 22.40 |
| 32 open shards, 0 workers | 152.31 | 22.40 |

The 2-worker setting is the better of these tested configurations. Earlier
first-read results and later warmed runs are not a controlled cold-cache
comparison; they do not establish that all acceleration is caused by the mapping
limit alone. Shared OS/filesystem caches were never forcibly evicted.

Final prebuild code also processed one actual Agibot episode (1,196 subtask
samples, current image inputs), publishing a new dataset-local fingerprint with
image dimensions and expanded preprocessing identity. 49 focused CPU tests passed
with eight GPU cases deselected, covering prepared-input ages, mmap retention,
cache corruption, CLI wiring and checkpoint resume; Ruff/syntax passed.

The stored feature contract was inspected: three independent camera arrays,
`(rows, history, 197, 768)` in the historical cache, preserving BF16 token bits.
`DiTPolicy.encode_vision_features` calls `encode_image_tokens` and reshapes only;
trainable pooling runs later in `build_vision_tokens`. The 197 positions are the
CLS token plus all 196 spatial patches exposed to the online DINO policy, not
pooled/mean embeddings. Internal backbone storage tokens are not part of that
policy interface, either online or cached.

## Warm cloud mapping comparison and recovery

A same-code repeat with the original two-mapping limit, after the 32-mapping
runs, measured **173.67 samples/s** (updates 6–100, batch64/workers2). The 32-mapping
run's 217.24 is about 25% higher. Logged losses match exactly, so this configuration
changes reading overhead rather than input order or model math. The trials were
sequential on shared infrastructure, not randomized repeated confidence bounds.

The 32-mapping configuration then completed 200 updates (over five training
epochs), saved the standard checkpoint and restored it in a fresh process for
50 more updates, reaching step250. The first recovery attempt was correctly
rejected because the validation driver changed val_every with the target step
count. Keeping val_every fixed corrected the driver; checkpoint checks were not
weakened and the failed log was retained.

From step25 through 200, allocated GPU memory was 14.815–14.840 GiB and reserved
memory 24.170 GiB. Main-process RSS was 4.211–4.534 GiB; aggregate worker RSS rose
as file-backed mappings populated (13.35–28.20 GiB), and includes shared mapped
pages rather than unique anonymous ownership. The container's estimated
anonymous/SHM/kernel footprint was 5.954–6.497 GiB, and SHM at sampled points was
at most 0.541 GiB. No memory.max/OOM/OOM-kill event counter increased during either
measured training segment. These are bounded multi-epoch checks, not proof of
multi-day stability; the rolling checkpoint remains at update200 because the
50-update extension did not reach another save interval.

See `cloud2-repeat.json`, `stability200.json`, `stability250.json` and
`run_stability.py`. Use `PYTHONPATH=src:.` for the stability driver so it can reuse
the existing benchmark/resource helpers. The canonical feature arrays stay on
the cloud filesystem throughout these tests.

## Cloud batch-size screening

With 32 retained mappings and two workers, the same real 2,482-sample historical
cache was trained for 100 updates at each larger batch:

| Batch | Samples/s, updates 6–100 | Peak allocated GPU GiB |
|---:|---:|---:|
| 64 | 217.24 | 22.40 |
| 128 | 271.68 | 30.64 |
| 256 | 278.80 | 45.56 |

B256 is the highest measured throughput, but only 2.6% faster than B128 while
using 14.91 GiB more peak GPU memory. B128 is the practical speed/memory choice
for this workload; the small speed difference is not established beyond run
variation. B512 was not attempted: extrapolating the observed B128→B256 memory
increase, with 10% headroom, predicts 82.93 GiB, exceeding the 68 GiB allocation
budget. No intentional OOM test was used. Eight A800 GPU parity tests passed for
cached/live loss, gradients and denoising across both encoders and history modes.

These measurements use full pre-pooling tokens on cloud storage. They are not
the theoretical fake-data maximum or directly comparable with a different action
horizon, mixture, image/history size or cold working set. See `cloud-b128.json`
and `cloud-b256.json` for the finite training/resource logs.
