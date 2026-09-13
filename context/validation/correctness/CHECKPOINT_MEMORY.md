# Checkpoint recovery memory

A trained full 2B checkpoint (11,748,176,486 bytes; 2,521 tensors) was loaded
in separate CPU processes on `js_dev_1`, using torch 2.7.1. Every tensor was
then accessed at 4 KiB strides. `/proc/self/status` measurements are in
[checkpoint-memory.json](checkpoint-memory.json).

| Additional resident bytes after accessing tensors | Ordinary load | Private mmap |
| --- | ---: | ---: |
| Anonymous | 11,756,576,768 | 5,332,992 |
| File-backed | 5,427,200 | 11,749,736,448 |

Private mmap avoids allocating a full anonymous checkpoint copy before worker
forks. Accessed file-backed pages can still occupy RAM and RSS; this is not a
claim that the whole checkpoint uses only 5 MB, nor a guarantee against OOM.
CPU optimizer updates use copy-on-write pages and do not modify the checkpoint.

Production resume uses private mappings for modern path-based checkpoints on
POSIX. Legacy serialization, seekable streams and non-POSIX platforms retain
the ordinary loader. The mapping default is restored after loading. Atomic
checkpoint saving is unchanged.

The probes ran alongside quality training, so their load times are not a fair
throughput comparison and are omitted. These intervals must not be used to
attribute training speed differences to an optimization. The ongoing paired
training queue uses its original loader in a separate checkout.

Validation covers CPU/CUDA next-update and RNG equality, uninterrupted versus
resumed epoch transitions, legacy/stream compatibility, and checkpoint byte
immutability after a CPU AdamW update (including a shared-mapping caller default).
