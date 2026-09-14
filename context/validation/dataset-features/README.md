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
