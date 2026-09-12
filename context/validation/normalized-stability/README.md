# Normalized mixed-data stability (2026-09-13)

On js_dev_2, one A800 80 GB trained the default 2,015.9M-parameter policy
on one Kai0 and one Agibot episode (3,095 samples total), batch 4, BF16,
online frozen DINO, 200 optimizer updates followed by checkpoint restore
and 5 further updates. The accompanying script computes experiment-only
normalization statistics and attaches them to the actual dataset loader.
It does not modify production dataset statistics.

This exercise exposed two failures: homogeneous batches in a mixed loader
needed padding to the model's declared dimensions (PR #20), and sparse
action coordinates with equal q01/q99 amplified rare values by about a
million. Newly computed statistics now opt into a min/max fallback for
degenerate quantile intervals; effectively constant coordinates use a
unit-width interval. Existing statistics retain their original mapping.
Normalization and its inverse use the same bounds, without clipping.

After both fixes, all 205 updates completed with finite loss. Logged loss
was 0.4141–1.5078, versus intermittent million-scale values before the
normalization fix. Allocated GPU memory after warmup ranged from
17,388.5928 to 17,388.5933 MiB. Restore resumed from step 200.
The script asserts finite loss, update counts, and less than 1 GiB of
post-warmup memory variation; it removes only its own large checkpoints.

This is a stability and restore smoke test, not a convergence or held-out
task-quality result. One episode per source is not representative enough
to produce production normalization statistics. Original logs and derived
statistics remain on the server rather than being published here.

Run from an isolated checkout with `checkpoints` and `datasets` available:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  python context/validation/normalized-stability/run_stability.py
```

The script's data and output paths describe the validation server; adjust
them for another machine. Do not run two copies against the same output.
