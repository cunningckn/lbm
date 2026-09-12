# Long training and closed-loop validation

## Real mixed-data stability and held-out episodes

On js_dev_2, the default 2,015.9M-parameter BF16 policy trained for 5,000
updates, then restored the standard checkpoint and completed 10 more updates.
Batch 4, zero workers, frozen online DINO, CLIP task vectors, 150 action
steps, max prefix 4. The run used the PR #27 implementation, before the
mathematically equivalent compact-prefix optimization.

Twenty episodes each from Kai0 and Agibot were split using seed 123: 16
training and four validation episodes per source. Record paths were asserted
disjoint. Normalization was computed only on training episodes. Training
contained 46,718 samples; a fixed 64-sample validation selection covered both
sources and all held-out episodes. Validation sampling noise was fixed.

| Update | Held-out reconstruction error |
| --- | --- |
| 500 | 0.19842 |
| 1000 | 0.18396 |
| 1500 | 0.14272 |
| 2000 | 0.14290 |
| 2500 | 0.15166 |
| 3000 | 0.13312 |
| 3500 | 0.12873 |
| 4000 | 0.14043 |
| 4500 | 0.11611 |
| 5000 | 0.12308 |

All training losses and gradient norms passed finite checks. Allocated GPU
memory at the forward hook after warmup ranged from 17,388.5928 to
17,388.5938 MiB; this is steady-state allocated memory, not peak reserved
memory. The curve improves overall but fluctuates; it does not prove
convergence or multi-day stability. Replay re-reads the first 5,000 batches
before continuing, so resume startup is expensive with live video decoding.

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  python context/validation/remaining/run_long_training.py \
  --data-root /data/prepared --output /scratch/long-heldout --steps 5000
```

## Actual LIBERO closed loop

Selected protocol: `libero_spatial`, task 0, pick up the black bowl between
the plate and ramekin and place it on the plate. The prepared dataset had
45 matching demonstrations, split deterministically into 36 training and nine
held-out episodes (3,636 / 851 samples). Training-only normalization and
frozen DINO feature caches were used.

The full 2,008.6M-parameter policy trained for 3,000 updates, batch 32, BF16,
AdamW at 1e-4 with the standard 1,000-step warmup; 50 action steps, prefix 4.
The final held-out reconstruction error was 0.2132 over 832 samples; the
runner dropped the final incomplete 19-sample validation batch. The resulting checkpoint
served live camera observations through the standard LBM HTTP policy server.

Ten official initial states (indices 0–9), simulator seed 7, 10 settling steps, maximum
220 control steps, replanning every five steps, 10 diffusion steps:
**8 successes / 10 trials**. Trials 2 and 3 reached the step limit; successful
trials took 75–90 steps. All returned actions were finite. No environment or
inference exception was counted as a normal failure. The environment closed
cleanly. This is a single-task smoke/quality test, not a full-suite score or
an estimate of general robot performance. Policy diffusion noise was not fixed
in this run, so the success count is not guaranteed to repeat exactly. The official initial states are
not claimed to be independent of dataset collection.

The existing simulator venv had a missing Python executable and incomplete
packages. An isolated Python 3.10 path plus offline packages restored it;
MuJoCo 3.12 failed during robot setup, while the project's locked MuJoCo
3.2.3 with robosuite 1.4.1 passed actual EGL rendering and physics steps.
The user's venv was preserved.

Reproduction (run from repository root, with matching local encoder assets):

```sh
PYTHONPATH=src python context/validation/remaining/build_libero_features.py train \
  --dataset libero --data-root /data/prepared --batch-size 32 --num-workers 4 \
  --no-mmap --output-dir /tmp/lbm-libero-features-train
PYTHONPATH=src python context/validation/remaining/build_libero_features.py val \
  --dataset libero --data-root /data/prepared --batch-size 32 --num-workers 4 \
  --no-mmap --output-dir /tmp/lbm-libero-features-val
python scripts/train.py --feature-cache /tmp/lbm-libero-features-train \
  --val-dataset /tmp/lbm-libero-features-val --batch-size 32 --num-workers 2 \
  --steps 3000 --log-every 100 --val-every 500 --val-batches 27 \
  --ckpt-every 3000 --output-dir /tmp/lbm-libero-policy
python scripts/serve_policy.py --ckpt /tmp/lbm-libero-policy/last.pt \
  --robot-type libero --norm-stats /tmp/lbm-libero-train-norm.json \
  --host 127.0.0.1 --port 8123
# In the Python 3.10 simulator environment, while the server is running:
MUJOCO_GL=egl python context/validation/remaining/eval_libero.py \
  --repo /path/to/lbm --port 8123 --trials 10 --output /scratch/libero-eval
```

Dataset-derived statistics, feature arrays, checkpoints and raw observations
remain on the server; this directory contains aggregate results and drivers.

Verified checkpoints are preserved under
`checkpoints/validation_20260913/{mixed,libero}/` on js_dev_2, including
normalization/configuration and the closed-loop aggregate result. The mixed
checkpoint is update 5,000; the subsequent 10-step resume verification did
not save another checkpoint. These ignored server artifacts are not in Git.

The mixed checkpoint was additionally loaded for both source adapters after
fixing inference IO adaptation. Actual observations produced finite
`(150, 14)` Kai0 and `(150, 22)` Agibot actions while retaining the checkpoint's
20-state/22-action model dimensions. Normalization happens before state
padding; outputs are trimmed to the source contract, and unused camera slots
are masked. This validates deployment shape compatibility, not physical
Kai0/Agibot task success.
