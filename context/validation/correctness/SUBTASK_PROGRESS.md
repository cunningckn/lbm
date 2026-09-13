# Subtask implementation progress — 2026-09-14

Work in progress on `codex/subtask-instructions`, based on PR44. No claim of
completed subtask training or historical-input support.

## Implemented and checked

- Agibot task annotations now match the numeric episode ID, accept `task_name`,
  and retain legacy task-wide dictionaries. Missing episode matches do not
  borrow another episode's instruction. Malformed or ambiguous annotations fail.
- Adapter scan revision is centralized in `CustomSpec`; Agibot revision 2
  rejects old cached instructions without silently launching a full rebuild.
- Records capture annotation-file digests from the same bytes that were parsed.
  Scan-cache loads check each distinct annotation source once, including missing
  annotations becoming available. Image/vector cache identities are unchanged.
- On js_dev_1, 37 annotation, scan-cache and custom-layout tests pass; Ruff passes.
- A bounded real scan of 20 Agibot episodes reads nonempty instructions in 20/20,
  compared with empty language in the 20 previously inspected cached records.
  These selected episodes share one task. This is not a corpus-wide audit.

## Remaining before PR and training

- Shared subtask interval/instruction contract and Galaxea per-frame task lookup;
  confirm Agibot endpoint semantics. Preserve source frame coordinates and
  explicitly handle gaps, overlap, action windows and historical boundaries.
- Conservative parent grouping for Agibot alpha/beta variants and subtask splits;
  keep train/validation normalization isolated.
- Include language/annotation changes in training and feature-cache provenance;
  invalidate/rebuild selected task embeddings, not the whole image cache.
- Actual short Agibot, Galaxea and mixed training, validation and restart tests.
- Full regression, five-PR normalization review (PR45 reaches the PR40 baseline
  threshold), self-review, CI and merge. Then historical visual/state work.

The server test checkout `/tmp/lbm-io-test` contains these working files; the
original server repository is preserved. Source data and checkpoints remain
private. No GPU training was started by this implementation step.

## Parent identity and replay fingerprint — 2026-09-14

Agibot records now carry a collection parent ID shared across alpha/beta paths.
Holdout groups parent IDs before random selection, retaining all variants on the
same side. Language and record metadata now contribute to the dataset replay
fingerprint, while annotation digests no longer change physical episode identity.
The focused split/recovery/annotation run passed 23 tests with one CUDA skip.
These changes are uncommitted and remain subject to final regression.

The publisher explicitly describes Alpha as a subset of Beta:
https://github.com/OpenDriveLab/AgiBot-World
This supports conservative grouping of matching collection episode IDs.
The historical `scripts/convert_to_lerobot.py` link now returns 404. The current
publisher README links to https://github.com/Tavish9/any4lerobot ; inspected
`agibot2lerobot/agibot_h5.py` preserves action_config metadata but does not
resolve endpoint inclusion. Do not present half-open endpoints as a verified
publisher convention without further evidence. Generic internal intervals can
be half-open, with a separately explicit source-boundary conversion policy.

## Subtask windows and real GPU validation — latest

Shared `custom/instructions.py`, CLI/DataConfig `instruction_mode`, adapter
read_subtasks, physical-coordinate sampling, bounded history/action windows,
selected-interval normalization, and train/val feature-cache mode/revision
checks are implemented. Galaxea scan revision is also 2. A real bilingual
Chinese@English instruction exceeded CLIP's 77-token length; the adapter now
selects the provided English suffix. Failure evidence remains in
`subtasks/attempts/clip-overflow`; the corrected queue is running.

Final CPU run: 531 passed, 27 skipped, 14 deselected in `/tmp/lbm-subtask-cpu`.
Ruff and syntax pass. Agibot completed ten updates and fresh-process recovery
to twenty. Galaxea and mixed are queued serially by
`/tmp/lbm-subtask-matrix.py`, PID 3803100 (read latest `subtasks/launch.json`).
Artifacts: `/mnt/kpfs/workspace/jinaoqun/Projects/lbm-validation-20260913/subtasks`.
Completion files, matrix.log and matrix-exit-code determine the remaining work.
Do not restart successful Agibot. `run_smoke.py` documents the deliberately short
full-size 2B protocol. No matched throughput or convergence claim follows.

Real cache audit: two episodes per source, six camera jobs each, all six ready
for each source, eighteen sampled frames per source decoded successfully.
This checks structural readiness and decode, not complete pixel/source parity.
Private aggregate evidence is `subtasks/cache-audit.json`.

Remaining: finish/review the GPU matrix, collect aggregate results, finalize
`context/validation/subtasks/README.md` and REVIEW.md, commit/PR45/CI/self-merge.
Then history visual/state implementation and matched short comparisons.
