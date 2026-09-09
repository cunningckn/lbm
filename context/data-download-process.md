# Data download / process scripts

## 2026-09-09 — Official URL + dump conversion

`scripts/data/process*` used to mean scan / FK / mmap / norm. That was the wrong
"preprocess". Process is now **official downloaded corpus → the dump layout LBM
already reads**. Some dumps need a step (Galaxea tar extract, AgiBot Alpha/Beta
nesting, RMBench HDF5→LeRobot, DAS sample HDF5→slim); the rest are already that
layout after download.

Download scripts always print a real URL (HF dataset page or EgoVerse S3 sync).
Empty `hfd` repo IDs are gone. Official snapshots land in `datasets/raw/<name>`;
`datasets/<name>` is the dump. `--link-local` is the cluster symlink shortcut.

Catalog: `scripts/data/catalog.py`. Do not fold FK/mmap/norm back into process.
