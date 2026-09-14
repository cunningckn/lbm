# Five-PR normalization review — 2026-09-14

Previous baseline: PR45. Covered changes: PR46–49 plus this validation/review PR50.
Base commit: 55b7f7892bb6b409a2d6a7522eaf02f3c98fedfa. The next review baseline
is PR50 once merged; do not advance it while the PR is still open.

- History defaults and CLI remain in config/train_cli; frame and timestamp
  selection share lbm.history with bounded storage and causal masks.
- Dataset registration remains adapter-driven. No dataset-name conditionals were
  added to production training for Agibot/Galaxea or LIBERO experiments.
- LIBERO installer/evaluator share the environment path and explicit lock.
  Evaluation now rejects invalid ranges/actions, preserves distinct reports,
  records exceptions separately and closes environments on failure.
- Feature-cache identity and verification are centralized in feature_cache_paths.
  The prebuild shell entrypoint delegates to the existing Python builder and
  shared sharded writer instead of defining another extraction pipeline.
- Review found that prepared BF16 training reduced historical time offsets to
  BF16 while live inputs retained FP32. cast_prepared_batch now centralizes the
  prepared-input conversion and preserves time precision; regression tests cover
  FP32 ages, BF16 feature/state tensors and boolean masks.
- Cache train/validation contracts now compare preprocessing/precision/text
  identity when present, while retaining legacy metadata compatibility.
- Shard mapping retention is bounded (1–64 per process), uses a shared training
  default and preserves mmap-backed payloads. Sampler order and math are unchanged;
  loss logs match across the tested mapping/worker settings. Production caches
  remain on cloud storage and retain complete pre-pooling token grids.
- Private dataset tensors, normalization arrays and checkpoints remain on the
  server. Review artifacts contain aggregate metrics and reproduction scripts.

Validation: PR49 final-head hosted CPU/static checks passed; 49 focused CPU tests
and eight actual GPU loss/gradient/denoising tests passed separately. Real cloud
prebuild/reuse, 100-step mapping/worker/batch comparisons, a 200-step cloud run and
fresh-process extension to250 completed. The recovery guard correctly rejected a
driver that changed validation frequency; fixing the driver preserved the guard.

Eight paired short Agibot/Galaxea history runs completed with matched initial/input
hashes and fixed held-out noise. Results reverse across seeds, so defaults remain
unchanged; see QUALITY.md. Historical LIBERO training/recovery/deployment evidence
is recorded separately. The prebuild shell wrapper is executable, and the README
quotes placeholder paths correctly and shows cloud mmap tuning.

Final merge remains gated on this PR's exact-head CI. Short training cannot prove
convergence, policy quality equivalence or multi-day resource stability.
