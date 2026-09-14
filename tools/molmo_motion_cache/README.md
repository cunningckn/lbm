# MolmoMotion mmap cache

This standalone package converts MolmoMotion-1M source shards into a portable,
index-driven numeric cache. It deliberately does not import the LBM training
package, Torch, or CUDA; conversion can run in a small CPU environment.

The first implemented and validated adapter is DROID. DROID source tracks are
paired *_2d.npz and *_3d.npz members inside completed tar archives, while
camera calibration is stored in *_cameras.json members. The builder:

- uses the official droid_split.json entries as the source of truth;
- ignores incomplete names such as .part and never reads them;
- flattens each variable (T, N, D) trajectory into shard-level contiguous
  .npy arrays, recording (row_offset, T, N) in Parquet;
- stores visibility masks separately instead of padding every trajectory to a
  global size;
- materializes numeric camera intrinsics/extrinsics once per clip;
- stores only relative runtime paths, so the cache can be copied to another
  host without rebuilding its indices.

It does not reconstruct DROID MP4s. The official release needs the upstream
DROID video corpus for that step, and no video data is fabricated by this
converter. Consequently, the current DROID pilot validates the trajectory
and camera path only.

## Run from the LBM checkout

Use the already prepared lightweight Python environment on tc_dev:

    export PYTHONPATH=/home/tione/workspace/kainingchen/lbm/tools/molmo_motion_cache/src
    PY=/home/tione/workspace/kainingchen/xprobot_flow/.venv/bin/python
    RAW=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m
    OUT=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1/pilots/droid-pilot-512

    $PY -m molmo_motion_cache build-droid \
      --source-root "$RAW" --output "$OUT" --limit 512 --shard-size 256 --checks 24

    $PY -m molmo_motion_cache verify \
      --source-root "$RAW" --output "$OUT" --checks 32 --verify-hashes

    $PY -m molmo_motion_cache benchmark \
      --source-root "$RAW" --output "$OUT" \
      --samples 512 --warmup 32 --frames 8 --points 32 --seed 20260914 \
      --report /home/tione/workspace/kainingchen/lbm/reports/molmo_motion_droid_pilot_benchmark.json

build-droid writes into a same-filesystem temporary directory and atomically
renames it into the requested output only after structural and source-value
checks pass. It refuses to overwrite an existing output directory.

## Output contract

    <output>/
      dataset.json
      build_stats.json
      verification.json
      clips.parquet
      objects.parquet
      tracks_index.parquet
      cameras_index.parquet
      provenance/source_manifest.parquet
      shards/droid/000000/
        points3d.npy
        points2d.npy
        visibility3d.npy
        visibility2d.npy
        camera_intrinsics_measured.npy
        camera_intrinsics_ds.npy
        camera_extrinsics.npy
        shard.json
      SHA256SUMS
      PILOT_READY.json

PILOT_READY.json is intentionally distinct from the eventual full-release
READY.json: a pilot is valid and portable but is not a claim that all
MolmoMotion subsets or reconstructed videos have been converted.
