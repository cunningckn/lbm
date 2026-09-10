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

```bash
bash simulation/libero/install_env.sh

# T1 — LBM checkpoint server
bash simulation/libero/eval_policy.sh ./checkpoints/<run>/<step>.pt

# T2 — sim client
bash simulation/libero/eval_env.sh --task-suite-name libero_spatial
```

Server cameras are `image` / `wrist_image` (agent view rotated 180° to match training). History, if any, comes from the checkpoint's `train_config.json` (`history_length` × `history_freq`), not client flags.
