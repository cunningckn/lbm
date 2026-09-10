# RMBench

[RMBench](https://rmbench.github.io/) convert + eval, wired to this LBM tree.

| Path | Role |
| --- | --- |
| `simulation/rmbench/` | Convert, eval client (`main.py`), sim venv |
| `simulation/lerobot/` | `uv` env for HDF5 → LeRobot |
| `third_party/rmbench/` | Sim, assets, raw demos |
| `datasets/rmbench/` | LeRobot dump used by `lbm.dataloader` |

## Convert (raw HDF5 → LeRobot)

```bash
# once
cd simulation/lerobot && uv sync

bash simulation/rmbench/convert_rmbench_data_to_lerobot.sh          # all 12 tasks → datasets/rmbench/
TASK_SET=m1 bash simulation/rmbench/convert_rmbench_data_to_lerobot.sh 50
# single task:
(cd simulation/lerobot && uv run python ../rmbench/convert_rmbench_data_to_lerobot.py cover_blocks 50)
```

Raw demos: `third_party/rmbench/data/<task>/demo_clean/`. Download if needed:

```bash
(cd third_party/rmbench && bash script/_download_assets.sh && bash script/_download_data.sh)
```

`datasets/rmbench` may already be a symlink to a pre-converted dump. Conversion refuses to clobber it unless you pass `--overwrite` or a new `--output-dir` / `OUTPUT_DIR`.

## Train (LBM)

```bash
DATASET=rmbench ./scripts/train.sh
# or
bash simulation/rmbench/finetune.sh
```

## Eval

The policy runs in the **LBM root** uv env (`scripts/serve_policy.py`). The sim client is a separate Python 3.11 venv (SAPIEN / CuRobo) and talks to it over HTTP.

```bash
bash simulation/rmbench/install_env.sh
```

Two terminals:

```bash
# T1 — LBM checkpoint server
bash simulation/rmbench/eval_policy.sh ./checkpoints/<run>/<step>.pt

# T2 — sim client
bash simulation/rmbench/eval_env.sh put_back_block
```

Cameras: `cam_high` / `cam_left_wrist` / `cam_right_wrist`. Image history, if the run was trained with `history_length` > 0, is applied on the server from `train_config.json`.

## Eval results

| Tasks | TMC | Pi0.5 (RMBench Paper) | Mem-0 | Pi0.5 (Short-Term) |
|---|---|---|---|---|
| Observe and Pick Up | M(1) | 9% | 4% | **32%** |
| Rearrange Blocks | M(1) | 13% | 89% | **90%** |
| Put Back Block | M(1) | 11% | 90% | **96%** |
| Swap Blocks | M(1) | 24% | 67% | **88%** |
| Swap T | M(1) | **15%** | 14% | 4% |
| **Average** | M(1) | 14.4% | 52.8% | **62%** |
| Battery Try | M(n) | 16% | **28%** | 0% |
| Blocks Ranking Try | M(n) | 6% | **18%** | 0% |
| Cover Blocks | M(n) | 0% | **68%** | 38% |
| Press Button | M(n) | 0% | 0% | 0% |
| **Average** | M(n) | 5.5% | **28.5%** | 9.5% |
| **Total Average** | --- | 10.4% | 42.0% | 38.7 |
