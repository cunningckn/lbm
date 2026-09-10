# LeRobot dataset conversion env

Standalone `uv` environment for converting simulation demos (HDF5 / RLDS) into [LeRobot](https://github.com/huggingface/lerobot) datasets.

Python **3.11** is required because `tensorflow-cpu==2.15.0` only ships cp311 wheels. Git pins:

- `lerobot` @ `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`
- `dlimp` @ `ad72ce3a9b414db2185bc0b38461d4101a65477a`

## Setup

```bash
cd simulation/lerobot
uv python install 3.11
uv sync
```

## Smoke check

```bash
uv run python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME; print(HF_LEROBOT_HOME)"
```

## Convert

HDF5 (RMBench) — raw demos in `lbm/third_party/rmbench/data/`, output `lbm/datasets/rmbench/`:

```bash
bash ../rmbench/convert_rmbench_data_to_lerobot.sh          # all 12 tasks, 50 eps
TASK_SET=m1 bash ../rmbench/convert_rmbench_data_to_lerobot.sh 50
uv run python ../rmbench/convert_rmbench_data_to_lerobot.py cover_blocks 50
```

RLDS (LIBERO) — output `lbm/datasets/libero/`:

```bash
bash ../libero/convert_libero_data_to_lerobot.sh /path/to/modified_libero_rlds
# or:
uv run python ../libero/convert_libero_data_to_lerobot.py --data-dir /path/to/modified_libero_rlds
```

Torch is the CPU wheel (dataset writing does not need CUDA). Then train with LBM:

```bash
cd ../..   # lbm root
DATASET=rmbench ./scripts/train.sh
DATASET=libero ./scripts/train.sh
```
