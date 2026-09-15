# MolmoMotion cache contracts

## Capabilities

Numeric schema is version 1. Publication contract 2 binds the release root to
component READY files, component checksum manifests and root metadata. An old
version-1 release can be audited/read but cannot automatically resume a build.
The completed numerical release is **annotations-only**. It is not a complete
multimodal training corpus. RGB pilot directories are separate publications.

## Numerical layout

Parquet indices identify sample, split, object/hand role, shard, row offset,
frame count T and point count K. NPY arrays contain contiguous flattened T*K
rows. Public readers reconstruct `(T,K,2)` and `(T,K,3)` arrays by mmap.
Generic adapters convert points to float32 and visibility to bool; original
dtypes are recorded in the independent semantic test report. Conversion is not
claimed lossless for arbitrary higher-precision source values.

Source 2D image coordinates retain the source image coordinate convention;
no new resizing is performed. 3D coordinate systems and units are inherited
from each source; there is **no common assumed metre/world-coordinate contract**.
Training code must select a compatible source coordinate system explicitly.

For a missing visibility field, the adapter derives a mask from finite
coordinates. This is availability, not a new observation of occlusion.
Existing visibility values are retained as booleans. The known zero-byte
MolmoSpaces 3D member becomes NaN points and false 3D visibility, with an explicit
source anomaly and availability label. These values are never valid geometry.
Xperience retains `trust_weights` as float32 and `keep_mask` as bool metadata.

Camera poses and intrinsics can be sparse: preserve the original integer frame
indices rather than indexing a pose array by the requested video frame directly.
EgoDex/YTVIS pose convention is recorded as camera-to-world; HD-EPIC/MolmoSpaces
as world-to-camera by the adapters. These labels have not been independently
validated by reprojection. DROID retains the selected source extrinsics and
measured intrinsics; its downsampled intrinsics use the existing principal-point
scaling convention, covered by a source-field comparison, not a calibration claim.

## RGB pilot

One source video/view is one pilot shard, deduplicated by video ID. Each has
`png.frames.bin`, `jpeg.frames.bin`, and `frames.npy`. The latter contains
uint64 offset/length pairs and frame ordinal, plus int64 original PTS.
`dataset.json` records the rational time base, annotation FPS, actual input
member SHA-256, code digest, codecs and alignment policy. PNG is exact relative
to decoded RGB24; JPEG quality 95 is lossy. Original dimensions are retained.

`strict-time` is the default and refuses mismatched PTS/FPS. The tested explicit
`frame-index` experiment pairs equal frame ordinals and retains both clocks;
it reports `time_semantics_verified=false`. It performs no resampling or
truncation. This is not permission to treat the two clocks as interchangeable.

`MMapMotionReader.get_selection` accepts explicit frame/point IDs, returns
those IDs and annotation times, and optionally RGB with original PTS/time base.
Missing requested RGB raises an error. Without RGB, state is annotations-only.
The public entry points reject missing READY, unknown schema and corrupt indices.
Builder-only staging validation has a separate private entry point.

## Integrity and resume

SHA256SUMS must contain unique canonical relative paths and lowercase SHA-256
digests. Missing/extra component files and symlinks are rejected. READY binds the
manifest; schema and index hashes are checked before public reads. Full array
content hashing is an explicit audit, not performed on each data request.

Build contracts bind the actual fixed-source file content digest, package code
digest, schema and parameters. Different parameters or legacy missing contracts
cause refusal. Legacy audit/export does not invent a historical build contract.
Component staging is verified before publication; failure staging is retained.

SHA-256 establishes internal consistency, not independent semantic correctness
or authenticity against an attacker who can replace all manifests and markers.
