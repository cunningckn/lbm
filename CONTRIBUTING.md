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
