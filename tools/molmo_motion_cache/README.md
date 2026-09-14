# MolmoMotion mmap cache

This standalone CPU package converts the complete published MolmoMotion-1M
snapshot into a portable, index-driven cache. It does not import LBM, Torch,
or CUDA. Runtime paths are relative to the delivery root, so the finished
directory can be rsynced from Tencent Cloud to Kingsoft Cloud without rebuilding.

## Published subset coverage

| Subset | Converted payload |
| --- | --- |
| DROID | 2D/3D tracks, visibility, camera calibration, split/caption/ranges |
| EgoDex | object and hand tracks, visibility, dynamic camera, metadata |
| HD-EPIC | object tracks, visibility, dynamic camera, metadata |
| MolmoSpaces | tracks, camera, videos and robot H5 assets, metadata |
| Xperience | object and hand tracks plus metadata; camera/RGB need gated upstream reconstruction |
| YTVIS | object tracks, visibility, dynamic camera, metadata |
| Stereo4D | all metadata and published track indices; numeric tracks/camera/RGB need upstream reconstruction |

The builder uses the official split JSON as its source of truth. It indexes
uncompressed tar headers once and reads NPZ members by byte offset, without
extracting a loose-file tree. Every numeric trajectory is normalized to
`(time, point, dimension)` and flattened into shard-level `.npy` arrays;
Parquet rows retain sample/object identity, shape, and offsets. JSON metadata
is normalized into Parquet/JSON delivery tables instead of being parsed in the
training hot path.

The source package does not include reconstructed RGB for DROID, EgoDex,
HD-EPIC, Xperience, YTVIS, or Stereo4D. The converter records that limitation
and never fabricates missing pixels or geometry.

## Tencent Cloud paths

```bash
cd /home/tione/workspace/kainingchen/lbm
export PYTHONPATH="$PWD/tools/molmo_motion_cache/src"
PY=/home/tione/workspace/kainingchen/xprobot_flow/.venv/bin/python
RAW=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m
OUT=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1
```

Validate all 82 files against the pinned Hugging Face local-dir manifest:

```bash
"$PY" -m molmo_motion_cache inspect-source \
  --source-root "$RAW" --verify-hashes --require-complete --workers 32
```

Run a seven-subset smoke build without touching the production output:

```bash
MODE=smoke SMOKE_OUTPUT=/tmp/molmo-motion-full-smoke \
  PYTHON_BIN="$PY" WORKERS=8 SHARD_SIZE=2 CHECKS=2 \
  bash scripts/tc_cloud/run_molmo_motion_full.sh
```

Run or resume the production build locally:

```bash
MODE=full PYTHON_BIN="$PY" WORKERS=100 SHARD_SIZE=256 CHECKS=32 \
  bash scripts/tc_cloud/run_molmo_motion_full.sh
```

Inspect the exact TI-ONE request, then submit a committed clean checkout:

```bash
"$PY" scripts/tc_cloud/submit_molmo_motion_full.py --dry-run
"$PY" scripts/tc_cloud/submit_molmo_motion_full.py --submit
```

The request reserves exactly 100 CPU cores and zero GPUs. Submission snapshots
the current Git commit under `<output>/_jobs/<job>/code`; credentials remain in
the external key file and are never copied into the snapshot or request audit.
The tc_dev default API proxy can be overridden with `TC_API_PROXY` or
`--api-proxy`.
Use `scripts/tc_cloud/monitor_molmo_motion_job.py --task-id <id>` for periodic
status checks; `--audit-log` appends a credential-free JSONL history.

After a cross-cloud copy, rehash every delivered component before publishing it:

```bash
"$PY" -m molmo_motion_cache verify-release \
  --output /path/to/copied/v1 --verify-files
```

## Safety and completion contract

- A source preflight checks path and byte-size completeness before conversion;
  production mode also checks every available LFS SHA-256.
- Each subset is written to a same-filesystem staging directory, verified
  against source values, and atomically renamed only after success.
- A completed subset has `READY.json`; a limited smoke result has
  `PILOT_READY.json`. Only a complete seven-subset release gets the top-level
  `READY.json`.
- A rerun reuses completed subset directories and refuses ambiguous partial
  targets. The output lock prevents concurrent writers.
- MolmoSpaces source MP4/H5 tar shards are hard-linked when source and output
  share a filesystem (copied otherwise), with portable tar-member offsets.
  `rsync` still transfers their bytes normally to another host.

## Output layout

```text
<output>/
  READY.json
  dataset.json
  source_preflight.json
  subsets/
    droid/
    egodex/
    hdepic/
    molmospaces/
    stereo4d/
    xperience/
    ytvis/
  assets/
    archives/molmospaces/{videos,robot_trajectories}/*.tar
    assets_index.parquet
    archive_manifest.parquet
    source_metadata/
  _jobs/<job>/{code,request.json,response.json,submission.json,build.log}
```

Each materialized subset contains Parquet indices, shard-level NPY arrays,
provenance, verification metadata, and checksums. Normal data loading needs
only the converted output; raw tar/NPZ/JSON is required only for source parity
checks and before/after benchmarks.

## Before/after benchmark

DROID uses the original `benchmark` command. Other materialized subsets use
`benchmark-generic`, for example:

```bash
"$PY" -m molmo_motion_cache benchmark-generic \
  --source-root "$RAW" --output "$OUT/subsets/egodex" --dataset egodex \
  --source-records-per-track-kind 32 --samples 256 --warmup 16 \
  --frames 8 --points 32 --workers 8 \
  --report reports/molmo_motion_full_egodex_benchmark.json
```

Both paths use identical sample/window requests, materialize the same arrays,
and require equal checksums. The reported result is a warm-cache,
single-process numeric-window microbenchmark; it does not claim the same
speedup for RGB decode, distributed training, or GPU transfer.
