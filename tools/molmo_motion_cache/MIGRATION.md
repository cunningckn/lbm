# Independent delivery and migration

The commands below operate on explicit new destinations. Never overwrite v1.
Set PYTHONPATH to this package's `src` or install its dependencies/package.

```bash
export PYTHONPATH=/path/to/lbm/tools/molmo_motion_cache/src
python -m molmo_motion_cache.delivery audit /path/to/v1/subsets/droid --verify-files
python -m molmo_motion_cache.delivery export-component /path/to/component /new/path/component
python -m molmo_motion_cache.delivery export-release /path/to/v1 /new/path/annotations-release
python -m molmo_motion_cache verify-release --output /new/path/annotations-release --verify-files
```

Export copies only listed component files and required markers/manifests.
Root capabilities explicitly say annotations-only. `_jobs`, pilots, partials,
logs and locks from the original root are excluded. Copying uses real file
copies, never source hardlinks or symlinks. A new staging directory is retained
on failure; an existing destination is rejected. Full export reads/hashes and
copies hundreds of GB; it has not been launched by the completion review.

After export, rsync/scp the standalone directory to an explicit destination,
then run verify-release with --verify-files. Do not infer success from file
counts or transfer exit status alone. Do not call manifest-only audit a full
content check. On the target, install numpy and pyarrow in an isolated environment.

## Actual small migration acceptance, 2026-09-15

Tencent source copy:
`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/completion-tests/droid-export-20260915`

Kingsoft target:
`/mnt/kpfs/workspace/jinaoqun/Projects/lbm-completion-tests-20260915/droid-export-20260915`

512 samples, 26 files, about 100 MiB. Both machines passed full component hash
checks and an isolated Python process read an 8-frame x 32-point window. The
process denies opens under the original Tencent source/v1 paths using an audit
hook, disables bytecode writing and verifies unchanged copy hashes afterward.
On Kingsoft those original paths are also absent. This is a real cross-cloud
pilot acceptance; it does not certify transfer of the full release.

The old Kingsoft project venv had a broken Python link. The system interpreter
was Python 3.13 and lacked pyarrow. A new `venv` under the test directory uses
system numpy plus pyarrow 23.0.1. The host could not reach PyPI; a matching
cp313/manylinux_2_28_x86_64 wheel was downloaded locally and copied over SSH,
then installed with --no-index. No existing project environment was modified.

```bash
/mnt/kpfs/workspace/jinaoqun/Projects/lbm-completion-tests-20260915/venv/bin/python -I \
  /mnt/kpfs/workspace/jinaoqun/Projects/lbm-completion-tests-20260915/relocation_check.py \
  /mnt/kpfs/workspace/jinaoqun/Projects/lbm-completion-tests-20260915/droid-export-20260915
```

## Unavailable upstream modalities

The fixed source README lists reconstruction scripts for each upstream corpus:

| Subset | Still required | Dependency |
| --- | --- | --- |
| DROID | RGB | original DROID videos and reconstruction recipe |
| EgoDex | RGB | Apple EgoDex videos |
| HD-EPIC | RGB | original HD-EPIC videos |
| YTVIS | RGB | YouTube-VIS 2021 source videos |
| Xperience | RGB and camera | gated ropedia-ai/xperience-10m access |
| Stereo4D | trajectories, camera, RGB | original Stereo4D data/reconstruction |

These are blocked for a full-modality release while the required files/access
are unavailable. Upstream licenses are per-subdataset, not inherited from one
global license. The current review has not obtained the missing source corpora
or validated their reconstruction. Follow the pinned source README and scripts;
do not create placeholder images/camera/trajectories and label them complete.

## JPEG224 joint pilot migration

`export_rgb224_pilot.py RAW RELEASE FOUR_VIDEO_RGB NEW_PACKAGE` builds a compact
numeric component for exactly those video IDs and independently copies the RGB
component. It refuses more than four videos and any existing destination.
This is a bounded acceptance fixture, not the full-release exporter. The package
contains relative `numeric/` and `rgb/` paths, geometry and both timing domains.

Copy the completed package to a new cloud directory, then run:

```bash
python -I verify_rgb224_migration.py "$PACKAGE" "$EXTERNAL_REPORT" \
  --deny "$ORIGINAL_RAW" --deny "$ORIGINAL_V1" --deny "$ORIGINAL_RGB"
```

Run on both clouds and compare each returned array SHA-256, including JPEG RGB,
transformed tracks/K and original poses. The checker audits every payload,
denies opening original roots, and checks that no package file was modified.
Tested Kingsoft reader environment differs in Python/NumPy from the builder;
build fingerprint is provenance, not a requirement to rebuild on migration.
