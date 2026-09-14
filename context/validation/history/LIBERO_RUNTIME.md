# LIBERO installation and entrypoint repair

The history validation exposed two installation problems and one CLI problem:

- LIBERO's upstream `setup.py` declares no runtime dependencies. Installing its
  editable package did not supply bddl/future and the related runtime packages.
- The installer used the uncompiled input file and could resolve MuJoCo 3.12.0,
  which failed during robosuite joint-reference setup. The already recorded
  MuJoCo 3.2.3 version passed the same real environment reset/step.
- Passing the evaluation function to the installed Tyro version exposed
  `--args.host`-style flags, while the shell entrypoint supplied `--host`.
  Parsing the `Args` dataclass directly restores the documented flat flags.

## Changes

Direct dependencies remain in `simulation/libero/requirements.in`; the lock is
regenerated for Linux x86_64 and Python 3.10, the interpreter actually used by the
client. Installation now uses that lock and the shared headless exclusion list.
It stops on dependency errors rather than installing a reduced fallback.

Installation and evaluation share `LIBERO_VENV`. An available Python 3.10 can be
reused without downloading another interpreter, and `LIBERO_PYTHON` allows an
explicit interpreter choice. Reinitialization retains existing environment
files. Activation instructions use the selected path. The clock probe also
locates LIBERO's nested source package through `LIBERO_ROOT`.

## Validation

- Two subprocess regression tests passed: a failing locked install stops before
  fallback/editable installation and preserves an existing marker; evaluation
  uses the selected environment and forwards its arguments.
- Resolution produced 69 locked packages. Re-resolving from cache reproduced the
  same lock byte for byte. Ruff, Python compilation and Bash syntax checks passed.
- The actual installer ran offline on js_dev_1 with system Python 3.10.12 in an
  isolated copy of the existing client environment and a separate LIBERO source
  copy. Existing large packages were reused; missing/different wheels were
  prepared locally and transferred via SCP. This was a repair/reinstall test,
  not an empty-environment download test.
- Metadata checks confirmed all 69 installed package versions match the lock.
  Repeating the installer with Python downloads disabled succeeded and retained
  a deliberately created marker file. The original environment/source were not
  changed by these tests.
- The real `eval_env.sh --help` exposes flat flags, and actual Tyro parsing
  accepted host, port, task suite, replan steps and trial count.
- `check_libero_clock.py` ran directly in the installed environment. A real
  MuJoCo step advanced time from 0 to 0.05000000000000003 seconds, rendered the
  observation and preserved its timestamp through the client codec. This is an
  environment/interface test, not a trained-policy success-rate measurement.

The committed lock intentionally excludes keyboard-teleoperation dependencies
for the offscreen client. The model/history training and numerical measurements
remain documented in [README.md](README.md).
