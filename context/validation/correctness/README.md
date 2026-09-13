# Full-model acceleration validation — in progress

This directory implements the protocol in [PROTOCOL.md](PROTOCOL.md).
The full-model numerical matrix is complete; see [NUMERICAL.md](NUMERICAL.md).
Training-quality and task-success comparisons are still pending. Do not infer
long-run equivalence or change defaults from the numerical results alone.

Prepared drivers:

- `benchmarks/numerical_parity.py`: trained full-model FP32/BF16 loss,
  gradient, update-vector and multi-step parameter comparisons; baseline,
  repeat, native split, compiled, fused and combined variants.
- `train_comparison.py`: production training with paired seeds, fixed held-out
  noise, per-source validation, checkpoint recovery and resource telemetry.
- `compare_training.py`: refuses incomplete runs, mismatched initial weights,
  random sequences, validation schedules or sample counts before comparison.
- `serve_paired.py` and `../remaining/eval_libero.py --independent-trials`:
  reset policy/simulator RNG separately for each official initial state;
  interrupted rollouts remain explicitly incomplete and record errors.

Run from the repository root with `PYTHONPATH=src:.`. Raw tensors, feature
caches, normalization statistics and checkpoints must stay outside Git.

Current GPU test environment: isolated `/tmp/lbm-correctness` on `js_dev_1`;
artifacts under
`/mnt/kpfs/workspace/jinaoqun/Projects/lbm-validation-20260913/correctness`.
The original server checkout is preserved. CPU regression on `js_dev_1` with CUDA hidden:
516 passed, 27 skipped, 14 integration tests deselected (2026-09-13). Targeted driver tests pass on `js_dev_1`, including independent trial RNG,
paired-contract rejection and Linux worker telemetry.

The user authorized `js_dev_1` as an alternative to busy `js_dev_2`. Its single
A800 80GB was idle before execution, with a 139 GiB container memory limit.
The numerical matrix runs sequentially; no simultaneous model comparisons.
Both precisions completed baseline, repeat, native, compiled, fused and combined
four-update comparisons. Paired training selects the combined candidate; BF16
multi-step update differences remain a review item.
Unbounded RAM/SHM exhaustion testing remains excluded because a writable bounded
child cgroup is unavailable.


Full-model recovery smoke on `js_dev_1`: a four-update combined run and a
fresh-process 2 + 2 update run match byte-for-byte across all 2,521 checkpoint
tensors (model, optimizer, scheduler and replay/RNG state), and their loss and
held-out logs match. The four-update baseline/combined smoke also has matching
initial weights, random sequences and validation counts. These short checks
validate the drivers; they do not establish long-run training quality.

The three-seed, two-dataset, two-arm 3,000-update queue is running serially on
the single GPU, with a fresh-process recovery at step 1,500 in each run.
`training/matrix-exit-code` is written only when the queue exits; a zero value
plus the paired reports is required before claiming completion. The simulator
has separately reset task 0 and produced finite observations from both cameras;
this environment smoke did not evaluate a policy.

Linux worker RSS is a sum over descendants and may count fork-shared pages
multiple times. Use container anonymous/SHM/kernel usage alongside it; neither
summed RSS nor reclaimable file cache alone establishes a memory leak. Early
first-stage logs lack per-process RSS; GPU/container/SHM telemetry is present,
and `/proc`-based RSS collection covers subsequent processes.
