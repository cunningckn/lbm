# Acceleration correctness and default-selection protocol

Baseline: PR #40 (`4e633ab`). Numerical parity precedes training and deployment
comparisons. All private tensor snapshots, data, normalization and checkpoints
stay on the test server; only aggregate metrics and drivers may be committed.

1. Full trained 2B mixed-data checkpoint, fixed real feature batches, fixed
   noise/timesteps/state masks/prefix RNG, FP32 and BF16 separately. Compare
   baseline repeat, native split, compilation alone, fused AdamW alone and the
   combination to the pre-split baseline. Report loss, all trainable gradient
   tensors, actual optimizer update vectors and parameter drift over four
   updates. Use trained weights to exercise nonzero gates, a fresh optimizer
   and LR 1e-4. Distinguish first-step parity from trajectory divergence.
2. Select a candidate using numerical evidence. Train baseline/candidate with
   paired seeds 123, 456, 789, identical source split/order/batch and 3,000 total
   updates on mixed data (20 episodes per source, 20% held out) and the
   prepared LIBERO task-0 dataset. Fixed-noise,
   per-source held-out reconstruction every 250 updates. A difference of more
   than 5% in paired final reconstruction is a review trigger, not automatic
   proof of a regression; report individual seeds and uncertainty.
3. Evaluate each LIBERO checkpoint on the same official initial states 0–49,
   simulator seed 7 + initial-state index, policy seed 2026 + index, 220 control steps, replanning every
   five steps and ten diffusion steps. Keep exception counts separate. Report
   paired successes/disagreements and uncertainty; task-0 results do not
   constitute a full-suite LIBERO score.
4. Monitor process/worker RSS, container memory, SHM, GPU allocated/reserved
   memory, finite updates, epoch transitions, validation and checkpoint I/O.
   Exercise fresh-process recovery. Add a sustained batch-128 candidate run
   only after its numerical gate passes. Never exhaust shared RAM/SHM without
   a writable bounded child cgroup. Shared container memory is not attributed
   solely to an experiment.
5. Decide defaults from the combined numerical, quality and stability evidence.
   Keep acceleration opt-in when evidence is inconclusive or a regression is
   unresolved. Record decisions and merge only after relevant tests and CI pass.

Numerical review thresholds fixed before full-model results: first-step loss
relative error 1e-4 (FP32), 1% (BF16); aggregate gradient relative L2 1e-4
(FP32), 2% (BF16); individual nontrivial gradient relative L2 1e-3 (FP32),
5% (BF16). Report update-vector error separately (review trigger 10%) because
Adam's near-zero gradient division and BF16 rounding can magnify it. Do not
substitute parameter-relative error for update-relative error. Report repeat
baseline noise and do not silently loosen thresholds after observing results.
