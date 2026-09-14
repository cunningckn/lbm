# Mix dumps: download + process

Training reads `datasets/<name>`. These scripts fetch the **official** corpus, then
(when needed) convert it into that layout.

```bash
# Hugging Face mirror (default). Override with HF_ENDPOINT=https://huggingface.co
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

./scripts/data/download.sh                       # every dump in DATASETS
DATASET=droid ./scripts/data/download.sh         # → datasets/raw/droid
DATASET=droid ./scripts/data/process.sh          # → datasets/droid
DATASET=galaxea ./scripts/data/download.sh       # lerobot/*.tar.gz only
DATASET=galaxea,kai0 ./scripts/data/process.sh
LINK_LOCAL=1 DATASET=droid ./scripts/data/download.sh
```

On `tc_dev`, the 50-CPU conversion wrapper defaults to
`/home/tione/workspace/kainingchen/Datasets`. It uses up to 48 mmap workers,
bounded by the CPUs available to the current job:

```bash
cd /home/tione/workspace/kainingchen/lbm
DATASET=droid CHECK_ONLY=1 ./scripts/data/convert_50cpu.sh
DATASET=droid SOURCE=/path/to/Droid/droid_1.0.1 ./scripts/data/convert_50cpu.sh
DATASET=interndata_a1 SOURCE=/path/to/InternData-A1 ./scripts/data/convert_50cpu.sh
```

Set `BUILD_MMAP=0` to create only the processed-dataset link, `WORKERS=N` to
change mmap parallelism, and `FORCE=1` only when an existing destination must
be replaced. The wrapper uses `/root/.local/bin/uv` as the `tc_dev` fallback
and requests Python 3.12 plus the `data` dependency extra. When package-network
access is unavailable, set `PYTHON_BIN=/path/to/a/prepared/python3.12` to use an
existing environment instead of `uv`.

`LINK_LOCAL=1` skips Hugging Face and symlinks a processed dump under
`/mnt/open_source_data` when that path exists.

`process` is **not** scan / FK / mmap / norm. Those stay:

- `scripts/prebuild_fk.py`
- `scripts/prebuild_mmap.py`
- `scripts/compute_norm.py`

| dump | official URL | process |
| --- | --- | --- |
| `abc` | https://huggingface.co/datasets/XDOF/ABC-130k | none (already MCAP) |
| `agibot` | https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha + [Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | nest as `AgiBotWorld_alpha` / `AgiBotWorld_beta` (native HDF5, not LeRobot) |
| `das_gripper` | https://huggingface.co/datasets/genrobot2025/DAS-Sample-Data (sample). Full MCAP: https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData | HDF5 sample → slim `episode.hdf5` + wrist mp4. 10Kh MCAP is not converted here. Cluster dump is already slim. |
| `droid` | https://huggingface.co/datasets/lerobot/droid_1.0.1 | none |
| `egoverse` | https://github.com/GaTech-RL2/EgoVerse — `python egomimic/scripts/data_download/sync_s3.py --local-dir DEST --filters aria-all` | none (S3 zarr is the dump; VRS→zarr is upstream) |
| `galaxea` | https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset (`lerobot/` only) | extract `<task>.tar.gz` |
| `hifi_umi` | https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K | none |
| `hy_lance` | https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data | none |
| `interndata_a1` | https://huggingface.co/datasets/InternRobotics/InternData-A1 | none (nested LeRobot v3 repositories) |
| `kai0` | https://huggingface.co/datasets/OpenDriveLab-org/Kai0 | none |
| `libero` | https://huggingface.co/datasets/physical-intelligence/libero | none (HF dump is already the LeRobot corpus) |
| `molmoact` | https://huggingface.co/datasets/allenai/MolmoAct-Dataset | none (nested household/tabletop LeRobot repositories) |
| `rmbench` | https://huggingface.co/datasets/TianxingChen/RMBench (`data/*/demo_clean/**`) | HDF5 demos → one LeRobot v2.1 repo |
| `robocoin` | https://huggingface.co/collections/RoboCOIN/robocoin | none (stage the gated per-task repositories under one directory) |
| `robotwin` | https://huggingface.co/datasets/lerobot/robotwin_unified | none. Official HDF5: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0 — convert with RoboTwin/XPolicyLab if you start from that. |

Gated HF repos need a token (`HF_TOKEN` / `hfd.sh --hf_token`). Destinations:

- official snapshot: `datasets/raw/<name>` (`LBM_DATA_RAW`)
- LBM dump: `datasets/<name>` (`LBM_DATASETS`)
