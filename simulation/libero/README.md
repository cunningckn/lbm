# LIBERO

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) convert + eval, wired to this LBM tree.

| Path | Role |
| --- | --- |
| `simulation/libero/` | Convert script, eval client (`main.py`) |
| `simulation/lerobot/` | `uv` env for RLDS → LeRobot |
| `third_party/libero/` | LIBERO sim / BDDL |
| `datasets/libero/` | LeRobot dump used by `lbm.dataloader` |

## Convert (RLDS → LeRobot)

Raw data: [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds).

```bash
# once
cd simulation/lerobot && uv sync

bash simulation/libero/convert_libero_data_to_lerobot.sh /path/to/modified_libero_rlds
# → lbm/datasets/libero/
```

Train:

```bash
DATASET=libero ./scripts/train.sh
```

`datasets/libero` may already be a symlink to a pre-converted dump (`/mnt/open_source_data/libero`). Conversion refuses to clobber it unless you pass `--overwrite` or a new `--output-dir`.

## Eval

The policy runs in the **LBM root** uv env. The sim client is a separate Python 3.10 venv (MuJoCo / robosuite) and talks to it over HTTP.

The evaluation lock targets **Linux x86_64 / Python 3.10**. Installation uses
`requirements.txt`, including the tested MuJoCo 3.2.3 version and LIBERO's
otherwise undeclared runtime dependencies. A dependency failure stops the
installer; it does not report success with a smaller substitute environment.

```bash
bash simulation/libero/install_env.sh

# T1 — LBM checkpoint server
bash simulation/libero/eval_policy.sh ./checkpoints/<run>/<step>.pt

# T2 — sim client
bash simulation/libero/eval_env.sh --task-suite-name libero_spatial
```

To select an existing interpreter or keep a separate environment:

```bash
export LIBERO_VENV=/path/to/libero-venv
LIBERO_PYTHON=/usr/bin/python3.10 UV_BIN=/root/.local/bin/uv \
  bash simulation/libero/install_env.sh
bash simulation/libero/eval_env.sh --task-suite-name libero_spatial
```

`LIBERO_VENV` is shared by installation and evaluation. Existing files in that
environment are retained while locked packages are installed. Without
`LIBERO_PYTHON`, uv reuses an available Python 3.10 and downloads a managed
interpreter only if needed. `UV_BIN` defaults to `uv` on PATH.

For an offline server, prepare compatible Linux/Python 3.10 wheels locally,
transfer the wheelhouse, and set `UV_OFFLINE=1` and `UV_FIND_LINKS=/path/to/wheels`.
Use an existing `LIBERO_PYTHON`; include built wheels for source-only packages
such as bddl, gym and future. A fresh environment needs all locked packages,
whereas a compatible existing environment can reuse its installed packages.

Maintain direct dependencies in `requirements.in` and regenerate the lock with
the command recorded at the top of `requirements.txt`. `headless-excludes.txt`
is shared by resolution and installation: keyboard teleoperation dependencies
are omitted from this offscreen client. Keep the MuJoCo pin unless a replacement
has passed real environment creation/reset/step tests with robosuite.

Server cameras are `image` / `wrist_image` (agent view rotated 180° to match training). History, if any, comes from the checkpoint's `train_config.json` (`history_length` × `history_freq`), not client flags.

### Bounded closed-loop checks

Select a small task/trial range before evaluating a whole suite:

```bash
bash simulation/libero/eval_env.sh --task-ids 0 --num-trials-per-task 2 \
  --init-offset 0 --no-save-video --result-path /tmp/libero-smoke/result.json
```

The result path must be new. Reports record completed trials, success counts,
server metadata and errors; inference or simulator errors exit unsuccessfully
and are not counted as ordinary task failures. Environments close on both
success and failure. Optional videos use a separate run directory and unique
task/trial names. Official initial-state ranges are checked before running.
Simulator seeds reset to `seed + initial_state` per trial; this does not fix the
policy's diffusion RNG. Use a separately seeded policy server for matched
comparisons. History receives timestamps and episode IDs at replanning points;
use `--replan-steps 1` when every control observation must reach the policy.
