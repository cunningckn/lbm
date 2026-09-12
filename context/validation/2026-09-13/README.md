# GPU and real-data validation — 2026-09-13 (Asia/Shanghai)

Source: `05d9233cc3458405af4062b3167b9175ab31aac0` (master after PR #15).
Runs used an isolated checkout at `/tmp/lbm-gpu-validation` on `js_dev_2`.
The server's modified checkout was not updated; its existing checkpoints and
datasets were linked into the isolated checkout.

## Environment

- Two idle NVIDIA A800-SXM4-80GB GPUs; driver 535.183.06.
- Python 3.12.13, PyTorch 2.7.1+cu126, CUDA runtime 12.6.
- `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src` for the commands below.
- Python/pytest executable: `/mnt/kpfs/workspace/jinaoqun/Projects/lbm/.venv/bin/`.

## Full integration result

**402 passed, 0 skipped, 0 failed**, 116.33 s. All GPU, real-data and
pretrained-asset prerequisites were enabled. Four warnings concern Lance/fork
compatibility; the two fork regression cases passed. This does not establish
that every untested production workload is free of bugs.

```bash
LBM_REAL_DATA=1 CUDA_VISIBLE_DEVICES=0,1 pytest -q -rs --require-pretrained-assets
```

[Full output](full-gpu-real.log). Targeted runs below are subsets of this suite,
not additional unique tests.

## Completed targeted tests

- GPU models, policy, checkpoint resume and distributed tests: **88 passed,
  zero skipped** (69.00 s). Includes real CLIP/DINOv3/SigLIP/T5 assets,
  CUDA RNG/resume, and a two-GPU FSDP forward/backward/optimizer step.
- Real dataset samples and FK: **16 passed, zero skipped** (57.76 s).
  All 11 adapters were exercised: abc, agibot, das_gripper, droid, egoverse,
  galaxea, hifi_umi, hy_lance, kai0, libero, rmbench.

```bash
CUDA_VISIBLE_DEVICES=0,1 pytest -vv --require-pretrained-assets tests/models tests/config/test_checkpoint_resume.py tests/distributed tests/policy
LBM_REAL_DATA=1 CUDA_VISIBLE_DEVICES='' pytest -vv -rs tests/dataloader/test_custom_real.py tests/test_fk_accuracy.py
```

## Synthetic single-GPU baseline

Default `DiTConfig`, BF16, attention auto, no compile, seed 0, five warmup
steps per phase and 30 measured steps. This uses synthetic inputs and does
not load pretrained weights. These are baselines, not measured improvements.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/train_throughput.py --device cuda --batch-sizes 1,4 --warmup 5 --iters 30 --json
```

| Batch | Compute samples/s | Including synthetic input preparation, samples/s | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| 1 | 7.19 | 5.81 | 18632.63 |
| 4 | 28.04 | 19.04 | 18635.24 |

Raw output: [synthetic-throughput.jsonl](synthetic-throughput.jsonl).
Input-preparation timing includes synchronization, not just CPU dataloader wait.

## Real Kai0 training smoke

[Script](real_train_smoke.py) and [raw output](real-train.log).
Run from the isolated checkout with `CUDA_VISIBLE_DEVICES=0`.

- Default training model: 2015.9M parameters, 1928.8M trainable; pretrained DINO.
- BF16, batch 4, two loader workers, one episode / 1800 samples, 20 updates.
- Three cameras, state/action dimensions 14; action window 5 seconds at 30 Hz.
- mmap disabled; output `/tmp/lbm-real-train-output`; no checkpoint/validation
  intervals reached and no external logging enabled.
- First five-step interval: 1.91 step/s, including startup.
- Subsequent intervals: 5.94, 6.03, 6.16 step/s (23.76–24.64 samples/s).
- Peak CUDA allocated memory: 18654.90 MiB; logged loss and gradient norms finite.

This confirms the real loading → model → backward → optimizer path. It is a
short, single-episode smoke run, not a convergence result or a statistically
robust throughput study. Kai0 norm_stats were absent: state/action stayed raw.
Its camera and temporal setup differs from the synthetic benchmark, so those
numbers should not be treated as a direct before/after comparison. FSDP was
validated for correctness; distributed throughput was not measured here.
