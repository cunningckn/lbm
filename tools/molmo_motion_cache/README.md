# MolmoMotion mmap cache

This standalone CPU package converts the complete published MolmoMotion-1M
snapshot into a portable, index-driven cache. It does not import LBM, Torch,
or CUDA. Runtime paths are relative to the delivery root, so the finished
directory can be rsynced from Tencent Cloud to Kingsoft Cloud without rebuilding.

## Code layout and compatibility

The package intentionally remains a small tool under `tools/molmo_motion_cache`.
Each source schema lives in `src/molmo_motion_cache/adapters/` (`egodex.py`,
`hdepic.py`, `molmospaces.py`, `xperience.py`, and `ytvis.py`); those modules
only locate and normalize source records. `generic.py` writes and verifies the
numeric shards, `generic_reader.py` reads finished shards without importing a
builder or consulting raw data, `rgb224.py` reads indexed JPEG payloads without
importing LBM, and `release.py` coordinates complete releases.
Existing `build-generic`, `build-release`, `verify-release`, and benchmark
commands keep their names and arguments.

New full releases record the immutable Hugging Face tree revision, a snapshot
fingerprint, and per-component fingerprints. This prevents a numeric
MolmoSpaces component from being paired with same-looking assets from another
source snapshot. The existing `v1` was created before this metadata existed:
it remains readable and its delivery manifests can be checked, but strict
cross-component pairing deliberately refuses it. Do not retrofit identity data
into `v1`; build a new output directory when strict pairing is required.

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

For a newly built, fingerprinted release, require source identity as well:

```bash
"$PY" -m molmo_motion_cache verify-release \
  --output /path/to/new-release --verify-files --require-source-identity
```

`--require-source-identity` is intentionally incompatible with the historical
`v1`; it fails closed instead of claiming that old components are safe to mix.

## Reader semantics that matter to training

- Xperience readers return `trust_weights` together with trajectories. They
  also return `source_point_indices`; it maps every retained cache point back
  to its original source point. `get_object_window()` returns both that mapping
  and `clip_frame_indices`, so point/frame selection cannot silently lose
  correspondence. For legacy Xperience rows, the reader derives the mapping
  from the stored `keep_mask`.
- MolmoSpaces `points3d` are in a fixed world frame and `camera_poses` are
  camera-to-world matrices. Projection must use `inverse(camera_poses)` as the
  world-to-camera transform. A build test projects a known sample to guard this
  convention; the converter does not flip the stored matrix direction.
- `MMapMotionReader(.../subsets/molmospaces).validate_paired_assets(.../assets)`
  checks completion manifests and source/component fingerprints before a new
  release joins numeric data with its assets.

## Safety and completion contract

- A source preflight checks path and byte-size completeness before conversion;
  production mode also checks every available LFS SHA-256.
- Each subset is written to a same-filesystem staging directory, verified
  against source values, and atomically renamed only after success.
- A completed subset has `READY.json`; a limited smoke result has
  `PILOT_READY.json`. Only a complete seven-subset release gets the top-level
  `READY.json`. The same marker creation and validation helper is used for
  numeric subsets, portable assets, and Stereo4D. Stereo4D is explicitly
  marked metadata-only; it does not imply that trajectories, camera, or RGB
  were reconstructed.
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

## RGB payload boundary

The current MolmoMotion cache release contains portable MP4/H5 tar assets, not
the `frames.npy` plus `frames-*.bin` JPEG layout, so `v1` cannot yet provide
RGB frames to a training Dataset. The refactor adds the independent reader for
that established future layout without modifying the existing image encoding:

- `frames.npy` is an `(N, 3)` integer matrix in `(shard, offset, length)` order,
  or a structured equivalent with those three names. Shards are named
  `frames-00000.bin`, `frames-00001.bin`, and so on.
- `Rgb224FrameReader` memmaps payload shards, returns a `memoryview` only in a
  scoped `borrow_jpeg()` context, and uses an LRU cap (`max_open_shards`) so a
  DataLoader worker does not accumulate file mappings. A forked or spawned
  worker reopens its own index and mappings.
- The old `seek/read` path remains available as `read_jpeg_bytes()` and
  `decode_rgb(..., mode="seek")`. The mmap path uses the same Pillow decoder;
  it avoids seek/read but does not claim zero-copy JPEG decoding.

Validate a future materialized payload before training:

```bash
RGB=/path/to/rgb224-payload
"$PY" -m molmo_motion_cache verify-rgb224 --payload-root "$RGB"
"$PY" -m molmo_motion_cache benchmark-rgb224 \
  --payload-root "$RGB" --samples 256 --warmup 16 --max-open-shards 8 \
  --report reports/molmo_motion_rgb224_benchmark.json
```

The benchmark checks equal JPEG bytes and equal Pillow RGB arrays before timing
the identical seek/read and mmap requests. It reports startup time, warm-cache
throughput, and p50/p95 latency; test `0`, `1`, and `4` DataLoader workers in
the actual training integration separately. The LBM training package keeps its
own mmap JPEG reader, but this cache package deliberately does not import it.
