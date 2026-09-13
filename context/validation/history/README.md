# Historical visual and state conditioning

This change adds opt-in timed visual history and state history to training,
feature caches and online inference. Defaults retain the old single-frame and
legacy history paths. It does not establish that history improves task success.

## Contract and compatibility

- `history_time_encoding=False` and `state_history_length=0` are the defaults.
  The state-history module has no parameters in a disabled checkpoint.
- Timed windows contain `max(1, round(length * frequency))` observations, including
  the present. Each target selects the nearest preceding observation. Episode and
  subtask boundaries are never crossed. Missing/stale history is masked.
- Physical-frame and timestamp sampling share `lbm.history`; actual relative
  ages enter the model. Epoch timestamps have explicit rounding tolerance.
- Invalid historical values and absent-camera values are sanitized before
  projections, and validity enters cross-attention. State masking also hides
  historical states. Current state remains separate for action transforms.
- State history uses a zero-gated residual attention pool, computed once per
  batch or denoising setup. Its initialization preserves shared weights and RNG.
- Online history requires observation timestamps in seconds. Context changes
  (episode ID, subtask ID, prompt) and explicit reset clear history. CPU snapshots
  contain resized images and copied states; storage and requested history are
  bounded to 64 observations. The HTTP server serializes access to its single
  policy context; it is not independent per-client session storage.
- Old feature-cache metadata receives disabled-history defaults. A requested
  history mode incompatible with the stored inputs requires a rebuild. History
  masks, relative ages and states survive feature-cache serialization/resume.
- LIBERO forwards its MuJoCo simulation clock. Other clients must supply their
  actual observation clock. Sending only replan observations does not supply
  intervening control frames.

## Verification so far

The final CPU suite passed **541 tests**, with 17 explicit skips and 46
GPU/integration deselections. Ruff and syntax checks pass.
Eight small-model A800 BF16 tests passed for live/cached loss, gradient and denoising parity
across DINO/SigLIP, one/two-observation history and both history modes.
A ninth GPU test verifies shared initialization and RNG preservation when the
model is constructed directly on CUDA.
A separate two-process comparison against frozen PR45 source confirmed exact
weight hashes, loss and three-step action equality for default single-frame and
legacy three-frame inputs on the small CPU model.

The cached training/resume test runs with both history modes enabled and
with both disabled. Other focused tests cover temporal order, invalid NaNs,
source masking, boundary resets, HTTP timestamp propagation and bounded storage.
These are scoped checks, not proof of arbitrary multi-GPU or long-run equivalence.

A real LIBERO task-0 physics step advanced the MuJoCo clock from 0 to
0.05000000000000003 seconds and preserved that timestamp through the client
codec. This checks the clock/interface, not historical-policy rollout success.
The shared environment had a missing Python 3.10 interpreter and undeclared
runtime dependencies, plus MuJoCo 3.12 drift. The check used the existing system
Python 3.10, isolated extra dependencies and **MuJoCo 3.2.3**, the version already
recorded in the project's requirements file. Server network failures were handled
by downloading/building small wheels locally and transferring them via SCP.
No original environment files were overwritten. The installer should separately
be made to install those runtime dependencies and honor the tested MuJoCo version.

## Real matched training

`run_matched.py` uses the production training loop, pretrained DINO/CLIP and the
full 2,037M-parameter policy on one A800 80GB. Each arm runs 20 updates on actual
Agibot + Galaxea subtask data, four episodes per source and a 25% parent-episode
holdout. Action duration is **one second** (30 padded action positions in the
mixture); history is three observations at 10Hz. No bulk/image cache build or
training checkpoint is requested. The script records finite losses, validation,
GPU peak allocation and initial/input/RNG hashes without publishing data arrays.

For each batch size, shared initial parameter hashes and the first three
state/action/language/CUDA-RNG hashes are checked across arms. Throughput uses
updates 6–20, excluding five warmup updates; timing is the production logger's
training-loop wall time. It excludes validation and setup. This short screening
window needs longer repeated measurements before making a production guarantee.

| Batch | Inputs | Samples/s | Peak allocated GiB |
|---:|---|---:|---:|
| 16 | Current | 57.10 | 18.45 |
| 16 | Visual history | 41.43 | 18.56 |
| 16 | State history | 55.99 | 18.52 |
| 16 | Both | 40.60 | 18.62 |
| 64 | Current | 143.81 | 21.60 |
| 64 | Visual history | 74.64 | 25.41 |
| 64 | State history | 142.66 | 21.66 |
| 64 | Both | 74.39 | 25.47 |
| 128 | Current | 173.08 | 28.78 |
| 128 | Both | 81.44 | 36.52 |
| 256 | Current | 162.33 | 41.78 |
| 256 | Both | 85.07 | 57.16 |

All three matrices exited successfully without OOM. Full evidence and paired
hashes are in `summary.json`. A proposed current-input B512 run was skipped:
the conservative projection exceeded the 68 GiB allocation budget. The measured
B256 current-input throughput was already below B128, so a larger allocation
was not justified by these results.

For this tested workload, **current input B128** is the fastest measured setting.
If both historical modalities are required, **B256** is the fastest measured
setting, but improves only about 4.5% over B128 while using about 20.6 GiB more
memory. B128 is a reasonable memory/throughput tradeoff for that mode. These
choices are specific to the measurements; no universal/global optimum is claimed.
Do not compare these one-second action-window figures directly with older
benchmarks using longer action chunks or different sources. Twenty updates do
not determine convergence or the best history configuration. Both new features
remain opt-in; the current-input default is preserved.

Reproduce an arm from the repository with existing data/cache/assets:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 python context/validation/history/run_matched.py \
  --mode both --batch 64 --steps 20 --output /new/private/output
```
