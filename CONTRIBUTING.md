# Contributing to LBM

Use Python 3.12 and start each change on a branch based on the latest `master`.
Keep each pull request focused on one fix or feature. Include its behavior,
validation results, and any remaining limitations in the description.

## Code checks

Install the development environment as described in the README. The Ruff
version is recorded in `uv.lock`; CI reads that version directly rather than
maintaining a second pin.

```bash
uv run --locked --extra dev ruff check .
python -m compileall -q src tests scripts examples benchmarks simulation
```

Ruff rules live in `pyproject.toml`. Imports, whitespace, line length and
undefined names are checked for repository Python code, including simulations.
Use `ruff check --fix` for safe automatic fixes and review the resulting diff.
CI runs lint and syntax checks on pull requests and pushes to `master`; these
checks do not install PyTorch, download weights, or require a GPU.

## Runtime validation

Run the tests that exercise the changed behavior in a suitable environment:

```bash
uv run --locked --extra dev --extra data pytest -q tests/config
uv run --locked --extra dev --extra data pytest -q tests/dataloader/test_mmap.py tests/dataloader/test_mmap_memory.py
```

These runtime tests are not yet part of hosted CI. Tests involving CUDA, real
datasets or pretrained weights require those resources to be available.
For training or performance changes, also record the model configuration,
batch size, hardware, warmup, measurement duration, peak memory and throughput.
Compare the same workload before and after the change.

## Adding a dataset

Add one adapter module under `src/lbm/dataloader/custom/datasets/` exposing
`NAME`, `SPEC`, `scan`, `read_vectors` and `read_frames`. The registry discovers
that module automatically; use `dataset_spec()` and `dataset_module()` from
`lbm.dataloader.custom` instead of adding name branches elsewhere. Add a focused
fixture/test and update a named mixture only when the dataset should be part of
that mixture.

## Training loader changes

`src/lbm/training_loader.py` owns train/validation DataLoader construction,
sampler selection, worker initialization and empty-loader checks. The runner
in `src/lbm/train_loop.py` owns dataset creation, epoch advancement and resume
replay. Keep loader construction changes in `make_loader()` and run
`tests/config/test_training_loader.py`, `tests/config/test_train_loop.py` and
`tests/config/test_checkpoint_resume.py` to check sampling and resume behavior.

## Pretrained assets in tests

Tests do not download CLIP assets implicitly. With missing CLIP or DINOv3
weights, dependent tests skip with the required paths; unrelated tests still
run. Supply CLIP model and BPE files under `LBM_CHECKPOINTS/clip` and set
`LBM_DINO` to the DINOv3 checkpoint to exercise pretrained tests.
For an environment expected to have those assets, use
`pytest --require-pretrained-assets` to make missing files a failure.
A run with skipped pretrained/GPU/data tests is not full integration coverage.
