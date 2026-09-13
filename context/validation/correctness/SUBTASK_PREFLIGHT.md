# Subtask preparation — bounded inspection, 2026-09-13

The current paired GPU experiment remains unchanged. This inspection reads
metadata and one sample table; it does not rebuild caches or claim a full audit.
Private schema results are stored on js_dev_1 under
`lbm-validation-20260913/correctness/subtask-preflight.json`.

## Findings

- Agibot's existing scan manifest declares 164,493 episodes / 291,255,488
  frames. Galaxea's declares 20,662 episodes / 26,358,560 frames. Both have
  the three expected scan-cache files and `complete: true`. These declarations
  do not establish source freshness, frame-cache integrity or annotation parity.
- Four sampled Agibot annotation files (two alpha, two beta) contain per-episode
  `episode_id`, `task_name`, and `label_info.action_config` segments with
  `start_frame`, `end_frame`, `action_text`, and `skill`.
- The existing Agibot `_instruction` reader takes the first list entry and only
  accepts `english`, `instruction`, or `task`. It misses the sampled schema's
  task name and segment text. The first 20 cached Agibot records all have empty
  language. This is a concrete instruction-ingestion issue; the sample is not
  a count of all affected episodes.
- One Galaxea episode has 728 frames, five distinct `task_index` values and
  four transitions, while `coarse_task_index` is constant. The shared LeRobot
  scanner currently chooses one episode-level language string, so it does not
  represent these per-frame instruction changes.

## Next implementation gates

1. Establish episode-ID matching, frame coordinate conventions, endpoint
   inclusion, missing labels, overlapping/gapped segments and fine/coarse task
   meanings from actual source annotations. Do not guess interval semantics.
2. Introduce a shared segment/instruction contract with source-specific readers.
   Preserve physical source coordinates for vectors and video caches; group all
   segments of one parent episode into the same train/validation split.
3. Include annotation identity and instruction mode in scan/feature cache
   compatibility. Corrected language requires regenerated task embeddings;
   unchanged image/vector caches may remain reusable after provenance checks.
   Rebuild only the selected test subset initially.
4. Test boundaries, normalization, action windows and historical inputs without
   leaking future frames or crossing the selected episode/subtask boundary.
5. Run real Agibot/Galaxea subtask training and held-out checks after the current
   GPU protocol completes, then implement/test the history-input ablations.

The existing acceleration comparisons still compare matching cached inputs,
but they do not demonstrate that Agibot language conditioning or subtask
semantics are correct. Report this limitation with the final quality evidence.

## Additional bounded inspection — 2026-09-14

Eight sampled Agibot annotations have adjacent segments whose next start equals
the preceding end. Some annotations omit initial/final episode frames; one
sample starts at zero and ends at `n_frames - 1`. These observations alone do
not resolve endpoint ownership. Define and document a deterministic boundary
policy after checking the source convention, rather than duplicating boundary
samples or assigning unlabeled tails silently.

Two matching task/episode IDs occur in both alpha and beta, with the same frame
counts and annotation boundaries. HDF5 file hashes differ; comparing 41 datasets
per pair finds differences in five end-effector/waist fields. These are not
byte-identical trajectories, but may be variants of the same collection episode.
Check canonical collection identity before splitting across alpha/beta; different
paths or file hashes alone are insufficient proof of independent episodes.
Private inspection results are `subtask-interval-sample.json`,
`subtask-duplicate-sample.json`, and `subtask-duplicate-values.json` under the
server's correctness artifact directory. No global duplicate rate is inferred.
