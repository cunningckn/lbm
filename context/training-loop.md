# LBM training loop

## 2026-09-01 — libero parquet keys; strict column names

Updated `datasets/libero` uses parquet columns `image`, `wrist_image`, `state`, `actions` (not `primary_image` / `observation.state` / `action`). Spec matches those names. LeRobot IO looks up spec keys exactly and raises `KeyError` listing available columns if a name is missing (no silent fallback / black frames). Parquet PNG/JPEG stills are decoded when the camera key is a parquet column.

## 2026-09-01 — mmap / norm tqdm

Progress bars go to stderr via `lbm.utils.progress.track` (on even when stdout is piped; off under pytest, `LBM_NO_PROGRESS`, DataLoader workers). Manual `prebuild_mmap` / rank-0 training prebuild: scan episodes, then `prebuild mmap-video`. Lazy auto-build of one missing JPEG pack: decode + jpeg bars. `compute_norm_stats` bars over episodes (`norm <dump>`).

## 2026-09-01 — mmap / norm preprocess scripts

`scripts/prebuild_mmap.py` and `scripts/compute_norm.py` reuse `load_dataset` + the same `--dataset` / `--data-mix` / `--data-root` / `--robot-type` flags as `train.py`. mmap prebuild is the rank-0 JPEG cache (`prebuild_mmap_caches`). norm writes `{norm_stats: {state, actions}}` JSON next to each dump (`<dump>/norm_stats.json`); a mix is one file per inner dump so dims stay compatible with `parse_norm_stats` / `load_norm_stats`. LeRobot `_vectors` reads parquet mmap when `use_mmap` is on (same path as training). `compute_norm.py --mmap` also prebuilds JPEG video in the same run (`MMAP=1` in `compute_norm.sh`). Shell wrappers: `prebuild_mmap.sh`, `compute_norm.sh`.

## 2026-09-01 — dataloader cleanup after GR00T removal

One loader family: `custom/datasets/<name>.py` auto-registers into `CUSTOM_SPECS`; catalog is generated from that. Task-folder mixes live in `custom/datasets/mixes.py`. `CustomMixtureDataset` is the only concat class (`ConcatMixtureDataset` is an alias). Dropped `custom/formats` and `readers_*` shims. `infer_spec_name` maps `meta/info.json` robot_type → spec.

## 2026-09-01 — libero / rmbench / robotwin are custom; GR00T and QwenVL removed

`libero`, `rmbench`, and `robotwin` are custom specs (`custom/datasets/<name>.py`). `--data-mix libero_all` and `--data-mix robotwin` expand task folders under data-root (`parse_mix("libero_all")` is `None`; `parse_mix("robotwin")` is the catalog entry, training still uses the folder mix). `embodiment_id` lives in `lbm.dataloader.embodiment`; `pad.py` no longer imports GR00T. `gr00t_lerobot`, `qwenvl_llavajson`, `vlm_datasets.py`, and `lerobot_datasets.py` are gone. mmap cache-build decode is sequential PyAV / OpenCV in `dataloader/mmap`.

## 2026-09-01 — custom native dumps use JPEG mmap

Agibot / DAS (HDF5+mp4), EgoVerse zarr stills, Hy Lance JPEGs, and ABC MCAP packets go through `lbm.dataloader.custom.mmap_video` into the same `.lbm_mmap/video` JPEG pack as LeRobot. `cam_high` on DAS stays a black placeholder (no file). numpy dumps stay in-memory.

## 2026-08-31 — mmap extracted; robotwin is GR00T; robodojo removed

JPEG / parquet mmap lives in `lbm.dataloader.mmap` (GR00T shims remain). Custom dumps use the same video cache: LeRobot / agibot / das MP4s via `read_mp4_all`, zarr / Lance stills and MCAP packets via `custom.mmap_video.decode_custom_mmap`. `robotwin` is catalog GR00T (Agilex); `--data-mix robotwin` still expands `DATASET_NAMED_MIXTURES`. RoboDojo custom spec is gone.

## 2026-08-31 — custom/datasets/<name> layout

Custom dumps are one module each under `lbm.dataloader.custom.datasets`. Shared LeRobot/numpy/JPEG helpers live in `custom/common/`. Add a dump by adding `datasets/<name>.py` with `SPEC` / `scan` / `read_vectors` / `read_frames` and registering it in `datasets/__init__.py`.

## 2026-08-31 — Named mix: all / custom_all / comma lists

`--data-mix all` concatenates every catalog folder (custom + libero + rmbench). `--data-mix custom_all` or `--data-mix kai0,galaxea` stay custom. Collate pads dims/cameras; `embodiment_id` is per sample. Sampling is concat-by-count, not equal weight per dataset.

## 2026-08-31 — Named datasets under ``lbm/datasets/``

On-disk dumps are linked at `lbm/datasets/<name>`. `lbm.dataloader.catalog` maps the name to backend (`custom` vs GR00T) and robot type; `paths.resolve_dataset` turns `--dataset kai0` into that folder. Custom `robotwin` / `robodojo` are gone from the custom backend. Catalog `robotwin` is GR00T Agilex (`--data-mix robotwin` / `--dataset robotwin`).

## 2026-08-31 — Custom dataset backend

`lbm.dataloader.custom` is a third loader family, independent of `gr00t_lerobot` and `qwenvl_llavajson`. Specs: agibot, galaxea, kai0, egoverse, das_gripper, hy_lance, hifi_umi, abc. Readers scan metadata only (nested LeRobot v2/v3, HDF5, zarr, Lance, MCAP, npz fallback). GR00T `robotwin` / `robotwin50` stay Agilex.

## 2026-08-31 — Dataset-agnostic train + heterogeneous pad

Training CLI is `scripts/train.py`. GR00T LeRobot via `--dataset` / `--robot-type` (rmbench, libero, …). Custom dumps (agibot, galaxea, kai0, …) go through `lbm.dataloader.custom`, not `gr00t_lerobot`. Collate pads action/state/T/cameras; `embodiment_id` is in the policy batch.

## 2026-08-28 — Frequency windows + full train/val loop

Training and the DiT policy are wired in `lbm.train_loop.main(TrainConfig)` (CLI: `scripts/train.py`). LeRobot batches are converted in `lbm.batch.policy_batch_from_loader`. Distributed: DDP by default under `torchrun`; `--fsdp` uses existing Megatron-FSDP.

Temporal windows are specified as **length (seconds) × frequency (Hz)**, not native-frame counts:

- `action_length` × `action_freq` → `chunk_length` (future action indices)
- `history_length` × `history_freq` → video history (past, including current). `history_length=0` is current frame only
- Native stride is `round(dataset_fps / freq)`

Defaults match the previous XL setup: 1 s × 50 Hz → 50 action steps; no image history.

## 2026-08-28 — Rank-0 mmap prebuild

Video JPEG mmap is built **before** training on rank 0 with a spawn process pool (`mmap_prebuild=True`, workers `min(32, CPU)`). After that `allow_build=False`, so DataLoader workers never decode AV1. Skip with `--no-mmap-prebuild`. Cache lives under the dataset at `.lbm_mmap/video/`.

## 2026-08-28 — Checkpoints live in `lbm/checkpoints`

Pretrained encoder downloads and loads (CLIP / DINOv3 / SigLIP / T5) default to `lbm/checkpoints/{clip,dinov3,siglip,t5}` instead of `~/.cache`. Training run checkpoints default to the same tree (`scripts/train.sh` → `checkpoints/`). Override with `LBM_CHECKPOINTS`, `LBM_CLIP` / `LBM_DINO` / … (lowercase `lbm_*` still works), or `--output-dir`. `~/.cache/<encoder>` is still accepted as a load fallback.
