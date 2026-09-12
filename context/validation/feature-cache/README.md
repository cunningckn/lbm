# Frozen vision feature caches

Precompute deterministic raw vision-backbone tokens and policy inputs:

```sh
uv run python scripts/prebuild_features.py --data-mix kai0,agibot \
  --data-root /data/prepared --batch-size 64 --num-workers 8 \
  --output-dir /scratch/lbm-features
uv run python scripts/train.py --feature-cache /scratch/lbm-features \
  --batch-size 64 --num-workers 2 --output-dir outputs/cached-training
```

Compute production normalization statistics before building the cache.
`--max-episodes` can cap a source for experiments. The destination must be a
new directory; failures leave no published partial cache. Arrays are written
in batches and read with mmap. BF16 fields preserve their bit patterns in
uint16 storage; other fields retain their dtype without quantization.

The cache contains raw backbone tokens, before the trainable vision pool.
It supports frozen DINO/SigLIP and `language_encoder=none` with CLIP task
vectors. Training the vision backbone or changing image preprocessing needs
live images and a rebuilt cache. Source observations and normalization are
snapshotted: editing the original data does not update an existing cache.
Encoder-weight/precision fingerprints are checked before training. The cache
owns camera, temporal and vector IO settings; DiT weights remain trainable.
Action-space definitions are preserved for inference. Validation must use
the same normalization and action-space contract as training.

Use a separately built held-out cache with `--val-dataset /path/to/cache` for
independent validation. Without it the runner evaluates the training cache,
which is not a held-out quality measurement. Standard checkpoint resume
checks the cache manifest identity as well as the normal replay signature.
Inference from trained checkpoints can still encode live camera images.

Validation covers live/cached loss, gradients and action sampling for both
vision backbones and history lengths 1/2, preserved camera masks, BF16
round-trip, interrupted writes, corrupt tables, encoder mismatch and training
with checkpoint restore. A real Kai0+Agibot cache was built from 11,829
samples across eight episodes using training-only normalization statistics.
Before the prefix projection optimization, batch 64/96/112 reached about
94/99/101 samples/s; batch 128 exceeded the A800 80 GB limit. Final tuning results are reported separately.
