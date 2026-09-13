# Trained full-model numerical comparison

Completed on the idle single A800 80GB in `js_dev_1`, using PyTorch 2.7.1+cu126.
See [the prewritten protocol](PROTOCOL.md) and
[scalar results](numerical-summary.json). These are numerical results, not a
completed training-quality or task-success evaluation.

The trained mixed-data step-5000 checkpoint exercises nonzero gates across
1,928,854,294 trainable parameters. Each precision uses the same initial weight
digest, real feature-cache batches (batch 4), noise, timesteps, state masks and
prefix RNG for four updates. AdamW starts with fresh moments at LR 1e-4.
FP32 matmul precision is `highest`. No dtype-to-dtype equivalence is claimed.

All first-step loss differences are zero. All first-step aggregate and
individual gradient and update metrics satisfy the prewritten review limits;
no zero-reference gradient tensor becomes nonzero. The largest first-step
individual BF16 gradient relative L2 error is 2.3943% (limit 5%).

| BF16 variant | First gradient error | First update error | Fourth gradient error | Fourth update error |
|---|---:|---:|---:|---:|
| Repeat baseline | 0% | 0% | 0% | 0% |
| Native prefix split | 0.2571% | 2.0392% | 0.9091% | 12.6494% |
| Compiled pointwise operations | 0.7374% | 7.7558% | 1.4504% | 14.2839% |
| Fused AdamW | 0% | 3.2261% | 0.7219% | 12.0561% |
| Combined | 0.7443% | 7.7604% | 1.5528% | 14.2279% |

Errors are relative L2 norms of gradients or **actual update vectors**. They are
not percentages of validation-error degradation. Later steps compare independently
evolving weights and optimizer states. Every BF16 optimization exceeds the 10%
update review trigger by step four. This remains an explicit trajectory-review
item for the paired training experiments; a small parameter-relative error must
not be used to dismiss it. Compiled-only step-three loss is 0.2333984375 versus
baseline 0.234375; equal rounded losses in other rows do not prove identical
training behavior.

In FP32, the largest fourth-step aggregate gradient error among the optimization
variants is 1.01e-6, and update error is 1.30e-5 (ratios, not percentages).
The repeat baseline itself has fourth-step gradient error 2.84e-7 and update
error 4.40e-6. This supports a low-precision numerical explanation for the larger
BF16 differences, without establishing long-run convergence or task equivalence.

Initial comparisons reduced FP64 statistics on CPU. Subsequent comparisons use
bounded FP64 chunks on the model device. Cross-device update statistics were
tested against CPU reductions with relative tolerance 1e-12 and absolute tolerance
1e-14. Reference snapshots were copied to local disk to avoid repeated shared
filesystem reads; raw snapshots remain outside Git.
An additional full-model BF16 repeat with the final GPU-reduction and atomic-report
implementation produced zero gradient and update differences for all four steps.

FP32 snapshot I/O reached the container's file-cache limit and caused reclamation.
No OOM or OOM kill occurred. Releasing only the unused experiment reference files'
page cache reduced container usage to approximately 40 GiB. This diagnostic I/O
phase is excluded from training-throughput claims. The cgroup filesystem is
read-only, so no unbounded memory-exhaustion test was attempted.

Decision for the next experiment: test `combined` against `baseline` with paired
training seeds, based on its first-step results and prior matched throughput.
Keep compile/fused settings opt-in while training-quality evidence is pending.
