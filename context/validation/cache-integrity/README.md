# Cache failure recovery

Changes reject truncated JPEG payloads, non-contiguous or invalid offset and
length tables, mismatched frame counts and non-UTF-8 manifests. Existing cache
formats remain supported; a failed readiness check uses the existing rebuild
path. Readiness validation accesses index tables, not JPEG payloads, with
bounded 65536-frame comparison chunks. Open also validates direct callers.
Table mappings are closed on failure and episode close.

Focused server regression: 47 passed (mmap, memory, integrity tests). Faults
include an actual SIGKILL of a writer after its temporary output exists,
ENOSPC injected at fsync, invalid UTF-8, truncated payload, wrong table dtype,
wrong counts and broken offsets. Retries produce readable ready caches.

`PYTHONPATH=src python rss_stress.py` in the server test checkout creates
262144 synthetic 4096-byte payloads (1024 MiB total) in a TemporaryDirectory.
It tests pack writing/index validation, not JPEG codec throughput. Process
peak RSS was 520.89 MiB including imported runtime; additional peak over the
pre-write baseline was **26.07 MiB**. The stress case asserts additional peak
below 128 MiB. Payload and indexes are removed on successful exit.

Full server regression with both A800 GPUs, real datasets and
`--require-pretrained-assets`: **416 passed, 0 skipped, 0 failed** in 189.24 s.
The four warnings remain the known Lance/fork compatibility warnings in fork
regression tests. This run includes MR #17 benchmark tests as well.
