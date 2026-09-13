# Full-model acceleration validation

The numerical, paired training, closed-loop and stability/recovery experiments
are complete. See [QUALITY.md](QUALITY.md) for results and limitations,
[NUMERICAL.md](NUMERICAL.md) for the full numerical matrix, and
[PROTOCOL.md](PROTOCOL.md) for the original protocol and user-requested
training-length amendment.

Compiled conditioning and fused AdamW remain opt-in: the short three-seed
closed-loop study does not establish quality equivalence. Private-mmap recovery
matches the uninterrupted reference byte-for-byte; see
[CHECKPOINT_MEMORY.md](CHECKPOINT_MEMORY.md).

[SUBTASK_PREFLIGHT.md](SUBTASK_PREFLIGHT.md) records concrete source-annotation
and cache concerns for the next subtask work. History-input implementation and
subtask training are separate pending tasks. Extreme RAM/SHM exhaustion remains
untested without an isolated bounded cgroup.

Run drivers from the repository root with `PYTHONPATH=src:.`. Publish only
aggregate metrics and reproduction code; raw tensors, features, normalization
statistics and checkpoints remain on the test server.
