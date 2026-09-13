# Paired acceleration quality and stability — 2026-09-14

**Decision: keep compiled conditioning and fused AdamW opt-in.** The full-model
numerical tests pass their first-update gates, and the tested runs are stable,
but three short training seeds do not establish quality equivalence. A lower
held-out reconstruction error does not necessarily improve closed-loop success.

## Protocol and scope

One A800-SXM4-80GB on js_dev_1, torch 2.7.1+cu126, BF16, full 2B policy.
Baseline uses unsplit prefix conditioning and ordinary AdamW; the candidate
combines native split-prefix, compiled conditioning and fused AdamW. The
combination comparison cannot attribute a quality change to one component.
Existing native split remains enabled; this report changes no runtime default.

All 12 training runs completed: mixed Kai0/Agibot uses three paired seeds and
3,000 updates; LIBERO task 0 uses the same three seeds and 1,500 updates. The
user requested the shorter LIBERO budget before any LIBERO arm started; see
[PROTOCOL.md](PROTOCOL.md). Batch is 32, validation every 250 updates, and every
run recovers in a fresh process halfway through. Initial weights, CUDA RNG
sequences, held-out schedules and sample counts match within each pair. The
original 1,000-step optimizer warmup remains; these runs are not convergence
or final-policy-capability evidence.

## Held-out reconstruction

Candidate-minus-baseline relative changes; negative means lower error.

| Source | Seed 123 | Seed 456 | Seed 789 | Mean | Descriptive 95% t interval over seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kai0 | -1.699% | +2.042% | -0.751% | -0.136% | [-4.967%, +4.695%] |
| Agibot | -0.134% | +3.642% | -1.974% | +0.511% | [-6.601%, +7.624%] |
| LIBERO | -2.256% | +0.398% | -6.466% | -2.775% | [-11.373%, +5.823%] |

LIBERO seed 789 triggers the unchanged 5% magnitude review threshold. Paired
input/RNG/count contracts pass, and the direction is lower reconstruction error.
However, its closed-loop success falls (below), so the trigger is resolved as
insufficient evidence for a default change, not waived as an improvement.
Only three training seeds are independent replicates in these descriptive
Student-t intervals; steps or repeated official states are not extra seeds.
No equivalence margin or final-success guarantee is established.

The Agibot instruction reader misses sampled real annotation fields and cached
language is empty in the inspected records; see [SUBTASK_PREFLIGHT.md](SUBTASK_PREFLIGHT.md).
Matched acceleration comparisons remain valid for those cached inputs but do
not validate correct Agibot subtask language conditioning. That is the next
implementation task, not silently repaired inside this experiment.

## Real LIBERO closed loop

`libero_spatial` task 0: move the black bowl between the plate and ramekin onto
the plate. Every checkpoint uses official initial states 0–49, simulator seed
7 + state index, diffusion seed 2026 + index, eager inference, ten diffusion
steps, replanning every five controls and at most 220 controls. All 300 trials
completed with finite actions and no rollout exceptions. This is a fixed
single-task comparison, not a full-suite or unseen-task score.

| Training seed | Baseline successes / 50 | Candidate / 50 | Baseline-only success | Candidate-only success |
| --- | ---: | ---: | ---: | ---: |
| 123 | 10 | 24 | 1 | 15 |
| 456 | 2 | 2 | 2 | 2 |
| 789 | 11 | 3 | 9 | 1 |

Success-rate differences are +28, 0 and -16 percentage points. Their mean is
+4 points, with a descriptive seed-level t interval [-51.32, +59.32] points.
The large between-seed variation and seed-789 decline prevent a claim that the
candidate is equally good or better. Short training may contribute to variation,
but this experiment does not isolate its cause. Do not compare these counts
causally with the older 3,000-step, differently seeded ten-trial smoke.

A controller port-probe failure occurred between the first and second arms;
no robot trial was running. It was fixed, failed-controller logs were archived,
and the completed first result was validated before reuse. See
[EXECUTION_NOTES.md](EXECUTION_NOTES.md).

## Stability and recovery

The combined candidate completed 1,000 real mixed-feature updates at batch 128,
including validation, epoch transitions, checkpoints and fresh-process recovery
at 500. Recorded GPU peak allocation was 63.22 GiB and peak reservation 64.63
GiB. Sampled container anonymous/SHM/kernel estimate peaked at 8.21 GiB; SHM
peaked at about 149.27 MiB. Recorded OOM and OOM-kill counters remained zero.
File-cache reclaim occurred; summed worker RSS includes shared pages and must
not be read as unique physical memory. Logs provide 24 resource snapshots, not
a continuous physical-memory peak measurement.

A separate full-model recovery using the new private-mmap checkpoint loader
resumed a two-step checkpoint for two more updates. All 2,521 checkpoint tensors,
optimizer/scheduler/replay state, initial-weight digest, training losses and
validation records match the uninterrupted four-step reference byte-for-byte.
The larger checkpoint-loading memory reduction is documented in
[CHECKPOINT_MEMORY.md](CHECKPOINT_MEMORY.md).

These stability runs do not establish a new throughput ranking. Dedicated
matched measurements and their scope remain in [../matched/README.md](../matched/README.md).
Batch 128 is the tested practical option with more memory headroom than 160;
compiled/fused acceleration still requires explicit selection. Destructive
RAM/SHM exhaustion remains untested because a writable isolated bounded child
cgroup is unavailable.

## Regression validation

The final CPU regression completed with 522 passed, 27 skipped and 14
deselected tests. Skipped or excluded cases are not claimed as integration
coverage; the real GPU experiments above provide separate evidence. Ruff and
Python syntax checks pass.

## Evidence and reproduction

- `training-summary.json`: complete per-seed validation curves.
- `quality-summary.json`: paired final quality, disagreements and intervals.
- `stability-summary.json`, `mmap-recovery-summary.json`: resource/recovery evidence.
- `train_comparison.py`, `compare_training.py`, `serve_paired.py`,
  `../remaining/eval_libero.py`, `summarize_quality.py`, `compare_checkpoints.py`:
  validation drivers. Run with `PYTHONPATH=src:.`.

Raw data, normalization, features and checkpoints stay on js_dev_1 under
`/mnt/kpfs/workspace/jinaoqun/Projects/lbm-validation-20260913/correctness`.
The original server checkout and its uncommitted changes are preserved.
