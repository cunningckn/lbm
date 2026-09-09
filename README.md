# LBM

Large Behavior Cloning Model.

## Install

Requires Python 3.12 and **PyTorch 2.7.1 + CUDA 12.6** wheels.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync --extra dev --extra data
```

Optional PyPI mirror:

```bash
# tsinghua
uv sync --extra dev -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# tencent
uv sync --extra dev -i https://mirrors.cloud.tencent.com/pypi/simple
```

Or from another project:

```bash
uv add /path/to/lbm
```

## Package layout

| Package | Contents |
| --- | --- |
| `lbm.config` | `DiTConfig`, `TrainConfig`, `ClipConfig`, `FlowConfig`, `OptimConfig` |
| `lbm.models` | `DiTPolicy`, DINOv3, CLIP text |
| `lbm.train_loop` | train / val / DDP / FSDP loop (`main(TrainConfig)`) |
| `lbm.utils` | preprocess, fake data, CUDA-graph inference, benches |
| `lbm.dataloader` | `custom` dump specs + `mmap`（JPEG / parquet 缓存）。训练在 `train_loop` 里组 loader。 |

## Dataloader (`lbm.dataloader`)

Episode parquet is converted once into `.mmap/` under the dataset root; subsequent reads use `numpy.load(..., mmap_mode="r")`. Video (MP4, HDF5-adjacent mp4, zarr/Lance JPEG stills, MCAP packets) is letterboxed to 224, JPEG-encoded (quality 85), and packed into mmap `frames.bin` under `.mmap/frames/episode_{id}/{camera_key}/`. The cache lives in `lbm.dataloader.mmap` and is used by every dump loader.

```bash
uv sync --extra data
```

```python
from torch.utils.data import DataLoader
from lbm.dataloader import collate_fn
from lbm.dataloader.custom import make_custom_dataset

dataset = make_custom_dataset("kai0", "kai0")
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=4)
```

Config knobs in `data_cfg`:

| Key | Default | Meaning |
| --- | --- | --- |
| `use_mmap` | `true` | mmap parquet proprio columns |
| `use_mmap_frames` | same as `use_mmap` | letterbox + JPEG pack + mmap for video frames |
| `image_size` | spec default (`224`) | square letterbox size before JPEG |
| `mmap_jpeg_quality` | `85` | JPEG quality |
| `action_length` | `1.0` | action window duration in seconds |
| `action_freq` | dump fps | action sampling frequency in Hz (`chunk_length = round(length × freq)`) |
| `history_length` | `0` | image history duration in seconds (`0` = current frame only) |
| `history_freq` | dump fps | image history sampling frequency in Hz |

Native-fps stride is `round(dataset_fps / freq)`. Disable mmap with `use_mmap: false` / `use_mmap_frames: false`.

| Module | Contents |
| --- | --- |
| `lbm.models.dit` | `DiTPolicy`, AdaLN-Zero blocks, `load_pretrained` |
| `lbm.models.dino` | DINOv3 ViT-B/16 (`DinoVisionBackbone`) |
| `lbm.models.siglip` | SigLIP ViT-B/16 (`SiglipVisionBackbone`) |
| `lbm.models.t5` | T5-small encoder (`T5LanguageEncoder`) |
| `lbm.models.clip` | CLIP ViT-B/32 text encoder |
| `lbm.models.attention` | flash / SDPA attention helper |
| `lbm.utils.preprocess` | state/action z-score, ImageNet resize-pad |
| `lbm.utils.fast_inference` | CUDA-graph `sample_actions` / RTC wrappers |
| `lbm.utils.fake_data` | synthetic batches / `FakeActionDataset` |
| `lbm.utils.bench` | `time_calls`, CUDA-graph capture for `sample_actions` |

Default `DiTConfig` (`hidden_size=1536`, `depth=32`, `num_heads=24`, `action_length=1s` × `action_freq=50Hz` → 50 action steps).

## Usage

```python
from lbm import DiTConfig, DiTPolicy, load_pretrained, make_fake_batch

model = DiTPolicy(DiTConfig())
load_pretrained(model, "checkpoint.pt")

# batch: state (B, 14), images {cam: (B, 3, 224, 224) or (B, T, 3, 224, 224)},
#        actions (B, chunk_length, 14), task_vec_clip (B, 512)
loss = model(batch)
actions = model.sample_actions(batch, num_steps=10)

# synthetic I/O, no dataset required
fake = make_fake_batch(DiTConfig(), batch_size=1)
```

## Training

`scripts/train.py` trains on a named folder under ``lbm/datasets/`` (`--dataset kai0`), a LeRobot path (`--dataset /path` + `--robot-type`), or a named mixture (`--data-mix` + optional `--data-root`). IO (cameras, state/action dims, chunk length) is inferred from the data. Heterogeneous mixtures pad action/state/T/cameras and carry `embodiment_id` in the batch.

```bash
# named dump under lbm/datasets/ (spec = catalog name)
uv run python scripts/train.py --dataset kai0
DATASET=galaxea ./scripts/train.sh

# LeRobot dumps with a spec (libero / rmbench / robotwin)
uv run python scripts/train.py --dataset rmbench
uv run python scripts/train.py --dataset /path/to/lerobot --robot-type libero

# mixture (default data-root is lbm/datasets)
DATASET=libero ./scripts/train.sh
DATA_MIX=agibot ./scripts/train.sh
DATA_MIX=robotwin ./scripts/train.sh

# merge several named dumps (pad action/state/T/cameras; embodiment_id per sample)
uv run python scripts/train.py --data-mix kai0,galaxea
uv run python scripts/train.py --data-mix all          # every named dump under datasets/

# DDP / FSDP
NPROC=8 ./scripts/train.sh --fsdp
```

Override temporal windows with `ACTION_FREQ` / `--action-freq` (default: each dataset's native fps). Training checkpoints default to ``lbm/checkpoints`` (`--output-dir` / `OUTPUT_DIR`). A `train_config.json` dump is written next to them.

Rank 0 prebuilds the JPEG mmap cache (``datasets/.../.mmap/frames``) with multiple processes before the first step; DataLoader workers only read it. Pass ``--no-mmap-prebuild`` to keep the old lazy build, or ``--mmap-prebuild-workers N`` to set process count. With mmap on, frames are already letterboxed. Pass ``--no-mmap`` to decode video live.

Preprocess without training (same ``--dataset`` / ``--data-mix`` / ``--data-root`` / ``--robot-type`` as ``train.py``):

```bash
# JPEG mmap under each dump's .mmap/frames/
uv run python scripts/prebuild_mmap.py --dataset kai0
uv run python scripts/prebuild_mmap.py --data-mix robotwin --workers 16
DATASET=kai0 ./scripts/prebuild_mmap.sh

# state/action mean/std/min/max/q01/q99 JSON next to each dump
uv run python scripts/compute_norm.py --dataset kai0          # datasets/kai0/norm_stats.json
uv run python scripts/compute_norm.py --dataset kai0 --mmap   # same + JPEG frame mmap
uv run python scripts/compute_norm.py --data-mix robotwin     # one file per task folder
DATASET=kai0 ./scripts/compute_norm.sh
MMAP=1 WORKERS=16 DATASET=kai0 ./scripts/compute_norm.sh
```

LeRobot proprio is read through parquet mmap (``.mmap/*.npy``) so stats match training. If a dump has no action columns, actions are the next proprio at ``round(native_fps / action_freq)``. Each spec's action space (GR00T-style ``rel`` arms / ``abs`` grippers; LIBERO stays absolute) is applied before stats and before train/infer norm: ``rel`` groups are current-state deltas, then ``[q01, q99]`` maps to ``[-1, 1]``. Inference inverts that after unnormalize. ``--mmap`` also runs the JPEG frame prebuild. A mix writes one ``norm_stats.json`` per inner dump (dims are not merged). Pass ``--output PATH`` only for a single dump.

`chunk_length` is derived: `round(action_length * action_freq)`. Image history similarly uses `history_length` × `history_freq`. Fake-data smoke test: `PYTHONPATH=src python examples/train.py`.

## Simulation eval (LIBERO / RMBench)

Convert dumps with `simulation/lerobot`. Train with the LBM root uv env. Closed-loop eval serves an LBM checkpoint (`scripts/serve_policy.py`); the sim venv POSTs `{state, images, prompt}` over HTTP and does not load LBM in-process.

```bash
# LIBERO
bash simulation/libero/eval_policy.sh ./checkpoints/<run>/<step>.pt
bash simulation/libero/eval_env.sh --task-suite-name libero_spatial

# RMBench
bash simulation/rmbench/eval_policy.sh ./checkpoints/<run>/<step>.pt
bash simulation/rmbench/eval_env.sh put_back_block
```

See `simulation/libero/README.md` and `simulation/rmbench/README.md`.

## 自定义数据集

官方下载和「官方格式 → 当前 dump」转换见 [`scripts/data/README.md`](scripts/data/README.md)（`download.sh` / `process.sh`）。scan / FK / mmap / norm 仍用 `scripts/prebuild_*.py` 和 `scripts/compute_norm.py`。

训练只有一套数据入口：`lbm.dataloader.custom`。每个 dump 一个 `CustomSpec`（`custom/datasets/<name>.py`），不需要 `modality.json`。加 spec 后在 `lbm/datasets/<name>` 放（或软链）数据即可。`--dataset kai0` 用 catalog 名当 spec；裸路径必须加 `--robot-type`。

`embodiment_id` 来自独立表 `lbm.dataloader.embodiment`（整数，须小于 32），`pad.py` 按 `robot_tag` 查表（样本上已有 id 则沿用）。

数据根目录默认是仓库内的 `datasets/`（可用 `LBM_DATASETS` 覆盖）。当前软链：

| 名字 | 磁盘 |
| --- | --- |
| `agibot` `galaxea` `kai0` `egoverse` `das_gripper` `hy_lance` `hifi_umi` `abc` | `datasets/<name>` |
| `libero` `rmbench` `robotwin` `droid` | `datasets/<name>`（含 `meta/info.json`） |

异构 mixture 仍 pad action / state / T / 相机。

### 1. 注册 dump

在 `src/lbm/dataloader/custom/datasets/<name>.py` 写 spec 和读盘函数（LeRobot 可复用 `common.lerobot`）：

```python
from lbm.dataloader.custom.common.lerobot import read_lerobot_frames, read_lerobot_vectors, scan_lerobot
from lbm.dataloader.custom.spec import make_spec

NAME = "my_robot"
SPEC = make_spec(
    "my_robot",
    "my_robot",
    ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    14,
    14,
    30.0,
    31,
    kind="lerobot",
)
scan = scan_lerobot
read_vectors = read_lerobot_vectors
read_frames = read_lerobot_frames
```

放进 `custom/datasets/<name>.py` 即可（`pkgutil` 会自动发现）。公共读写在 `custom/common/`（LeRobot / numpy / JPEG）。

已有 spec（相机 / state×action / fps / id）：

| spec | 本体 | 相机 | dim | fps | id |
| --- | --- | --- | --- | --- | --- |
| `agibot` | AgiBot | `top_head`, `hand_*` | 20 / 22 | 30 | 26 |
| `galaxea` | Galaxea | `head_rgb`, `*_wrist_rgb` | 32 / 14 | 15 | 11 |
| `kai0` | ALOHA | `top_head`, `hand_*` | 14 / 14 | 30 | 7 |
| `egoverse` | EgoVerse | `cam_high` + wrists | 16 / 16 | 30 | 12 |
| `das_gripper` | DAS | `cam_high` + wrists | 16 / 16 | 30 | 13 |
| `droid` | Franka (OXE) | `exterior_*_left`, `wrist_left` | 8 / 8 | 15 | 17 |
| `hy_lance` | Hy UMI | `cam_*` | 16 / 16 | 30 | 14 |
| `hifi_umi` | HiFi UMI | `head_main`, `*_hand_up` | 20 / 20 | 25 | 16 |
| `abc` | YAM | `cam_high` + wrists | 14 / 14 | 10 | 20 |
| `libero` | Franka | `image`, `wrist_image` | 8 / 7 | 10 | 25 |
| `rmbench` | ALOHA | `cam_high` + wrists | 14 / 14 | 50 | 7 |
| `robotwin` | ALOHA | `cam_high` + wrists | 14 / 14 | 30 | 7 |

`--dataset robotwin` 读 `datasets/robotwin` 这一棵树；`--data-mix robotwin` 展开 data-root 下约 50 个任务目录。LIBERO 转换写成一份合并 dump，用 `--dataset libero`。RoboDojo 已移除。

### 2. 磁盘格式

**numpy**（最简单）：

```
my_dataset/
  episode_000.npz   # state (T, Ds), action (T, Da), lang, image.<cam> (T, H, W, 3)
```

`make_custom_dataset` 只扫元数据，向量和视频在 `__getitem__` 里按 episode 读。各 spec 对应的真实 dump：

| spec | 格式 | 磁盘要点 |
| --- | --- | --- |
| `kai0` / `galaxea` / `droid` | LeRobot v2.x | 可嵌套 `task/.../meta/info.json`。galaxea 是拆开的 `observation.state.*` / `action.*` 列；droid 用 packed `observation.state` / `action`（8-D 关节+夹爪） |
| `hifi_umi` | LeRobot v3 packed | `meta/episodes/*.parquet` + `data/chunk-*/file-*.parquet`；hifi 在 `chunk-*/part-*/` |
| `agibot` | HDF5 + mp4 | `proprio_stats/{task}/{ep}/proprio_stats.h5`，视频 `observations/.../videos/*_color.mp4` |
| `das_gripper` | HDF5 + mp4 | 顶层 `*/das_gripper_slim_meta.json` 列 `episode.hdf5`（深度不一，勿 walk `[STAGE 3]`）；腕部 `cam_*_wrist.mp4`；缺的相机（含 `cam_high`）用黑帧 + `camera_mask` |
| `egoverse` | zarr | `*.zarr` 里 `left/right.obs_ee_pose` + JPEG `images.front_1`（只给 `cam_high`；腕部黑帧 + mask） |
| `hy_lance` | Lance | `table_*/table_*.lance` + `meta/hy_episodes.jsonl`；2 维 action 当作夹爪，EE 用下一帧 state |
| `abc` | MCAP | `data/train/<task>/episode_*/episode.mcap` |
| numpy | `.npz` | 任意 spec 在扫不到原生 dump 时回退 |

缺相机用黑帧并打 `camera_mask=False`，不会复制其他路的图。依赖：`uv sync --extra data`（含 `h5py` / `zarr` / `mcap` / `pylance`）。

### 3. 训练

```bash
uv run python scripts/train.py --dataset kai0
uv run python scripts/train.py --dataset galaxea
uv run python scripts/train.py --data-mix robotwin
DATA_MIX=agibot ./scripts/train.sh
```

绝对路径仍然可用。新 mix 加在 `custom/datasets/mixes.py` 的 `NAMED_MIXES`。

合并训练：`--data-mix kai0,galaxea` 或 `--data-mix all`（`custom_all` / `all_custom` / `all_data` 是同一 mix 的别名）。collate 按 batch pad 到最大 action/state 维和相机并集，`embodiment_id` 区分本体。采样是按样本数拼接（数据量大的 dump 步数更多），不是按数据集等权。`all` 里缺盘的成员会跳过；显式名单缺盘则报错。`--data-mix robotwin` 是 data-root 下的任务 mix，和 `datasets/robotwin` 这一棵树不是一回事。

## Examples (inference)

```bash
PYTHONPATH=src python examples/infer.py
PYTHONPATH=src python examples/infer.py --diffusion-steps 10
```

## FSDP

Native PyTorch DiT, wrapped with standalone `megatron-fsdp`. FSDP shards last
on a dense DP mesh. Synthetic-only FSDP smoke test:

```bash
torchrun --standalone --nproc_per_node=8 examples/train_fsdp.py
```

## Benchmarks

Synthetic data. Inference is closed-loop (`bs=1` only). Profile scripts are the
optimization signal: CUDA-event fwd/bwd/step plus wrap/prefetch sweeps.

```bash
# 1-GPU train throughput
PYTHONPATH=src python benchmarks/throughput.py
PYTHONPATH=src python benchmarks/throughput.py --batch-sizes 1,2,4

# inference Hz (bs=1): eager + CUDA-graph
PYTHONPATH=src python benchmarks/infer_hz.py
PYTHONPATH=src python benchmarks/infer_hz.py --paths eager,compile,cuda-graph

# 1-GPU where-the-time-goes (vision vs DiT vs Adam)
PYTHONPATH=src python benchmarks/profile_split.py --batch-size 32
PYTHONPATH=src python benchmarks/profile_split.py --batch-size 32 --kernels

# FSDP train; --profile prints fwd/bwd/optimizer ms
torchrun --standalone --nproc_per_node=8 benchmarks/throughput_fsdp.py --batch-size 32 --profile

# weak-scaling check (re-profiles on fail)
PYTHONPATH=src python benchmarks/scale_fsdp.py --world-sizes 1,2,4,8 --batch-size 32

# FSDP wrap / prefetch sweeps (always profiles)
PYTHONPATH=src python benchmarks/profile_fsdp.py --phase inventory
PYTHONPATH=src python benchmarks/profile_fsdp.py --phase all --nproc 8
```

## Tests

```
tests/
  conftest.py helpers.py fixtures/
  config/         # DiT / TrainConfig, temporal windows
  dataloader/     # collate pad, mixture, mmap, transforms, policy batch
  policy/         # checkpoint config, HTTP infer codec
  models/         # DiT, encoders, action mask
  distributed/    # FSDP wrap (incl. 2-GPU train step)
  utils/          # fake data, preprocess, benches
```

```bash
uv run pytest tests -q
uv run pytest tests/distributed/test_fsdp.py::test_fsdp_two_gpu_train_step
```

## Notes

DINO weights are **not** bundled. Follow the DINOv3 license, download `dinov3_vitb16_pretrain_lvd1689m.pth`, and put it in `checkpoints/dinov3/` (or set `LBM_DINO` / `lbm_DINO`). CLIP ViT-B/32 text weights download on first `CLIPTextEmbedder` / `--language-encoder clip` use into `checkpoints/clip/`.

SigLIP-B (`google/siglip-base-patch16-224`) and t5-small download to `checkpoints/siglip` / `checkpoints/t5` on first `--vision-encoder siglip` / `--language-encoder t5` train run. Override with `LBM_SIGLIP` / `LBM_T5` (aliases `lbm_SIGLIP` / `lbm_T5`), or `HF_ENDPOINT` / `lbm_SIGLIP_URL` / `lbm_T5_URL`. The whole tree is `lbm/checkpoints` (`LBM_CHECKPOINTS` to relocate). Pass `--no-pretrained-encoders` to keep random init. Vision and language towers are frozen by default; pass `--train-vision-encoder` / `--train-language-encoder` to train them. Attention-pool and proj layers are always trained.
