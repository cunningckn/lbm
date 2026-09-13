# Five-PR normalization review — 2026-09-14

Review baseline: PR40. Covered merged changes: PR41–44, plus the current subtask
change. Base commit: `6f69c062296c627669cd8cd9cece163f86df61ed`.
The next baseline is this subtask PR once merged; until then this review is
pending final GPU results and CI.

## Scope and findings

- Dataset registration remains automatic. Optional `read_subtasks` functions
  live in Agibot/Galaxea adapters. Interval validation, physical-coordinate
  lookup, gap exclusion and sampling counts are centralized in `instructions`.
  No dataset-name cases were added to the training loop or shared sampler.
- Instruction selection has one DataConfig default and one CLI binding. Old
  episode mode remains selectable. Scan revisions live with the dataset spec;
  both scan indexes and training/validation feature caches check corrections.
- The review identified mutable annotation content in episode identity and
  absent language in replay fingerprints. Parent grouping now keeps Agibot
  variants together; content changes invalidate replay without changing parent
  identity. Automatic holdout fits normalization only from training parents.
- Selected subtask ranges drive normalization, action-window clamps and image
  history clamps. Gaps are excluded without a frame-sized Python index. Source
  records keep full physical lengths, preserving vector, video and FK keys.
- Real Galaxea training exposed the bilingual text overflow. The adapter now
  selects the source-provided English component; failed evidence is preserved.
  CLIP token limits and model architecture are unchanged.
- Error paths include malformed/ambiguous annotations, invalid intervals,
  unknown task IDs, noncontiguous source rows, stale caches, unsupported subtask
  adapters and fewer than two independent holdout parents.
- Runtime defaults for compilation and fused AdamW remain opt-in. The previous
  full-model quality and private-mmap reports are retained; no old performance
  measurements are relabeled as subtask results.

## Compatibility and limits

Model state-dict keys and physical image/vector cache coordinates are unchanged.
Old exact-resume contracts do not silently accept changed language/data
fingerprints. Fresh-process recovery is tested for the new subtask protocol.
The raw validation runner disables bulk and image-cache builds; selected
parquet vector caches may populate lazily. No destructive memory-pressure test
is run on a shared server without a bounded writable child cgroup.

Final CPU regression: **531 passed, 27 skipped, 14 deselected**. Ruff and syntax
checks pass. The skips are not claimed as full integration coverage. GPU
outcomes and cache sampling are reported separately in README.md and the final
aggregate evidence. This review covers the changed paths, not a proof that the
entire project is free of unknown bugs.
