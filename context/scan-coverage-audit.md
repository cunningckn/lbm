# Scan coverage audit (dir_depth / discovery miss)

## 2026-09-09

Question: does any mix dump besides `das_gripper` drop most of the official corpus because a fixed `dir_depth` (or LeRobot `max_depth=5`) only indexes a shallow slice?

Method: compare `{dump}/.cache/episodes/manifest.json` to official metas / `meta/info.json` / one-level layouts. No `os.walk` on DAS `[STAGE 3]`, ABC `.cache`, or AgiBot trees.

`DATA_MIX=all` names: abc, agibot, das_gripper, droid, egoverse, galaxea, hifi_umi, hy_lance, kai0, libero, rmbench, robotwin.

### Verdict

**Only `das_gripper` has a real scan-miss.** Other linked dumps match their on-disk official counts (or drop 2 empty EgoVerse zarrs). `robotwin` is in the mix but has no `datasets/robotwin` symlink.

| dump | scanner limit | cache eps | official / on-disk | hours (cache) | miss? |
|---|---|---|---|---|---|
| das_gripper | `episode.hdf5` `dir_depth=3` | 12,985 | 667,804 (6 task metas) | 192 | **yes — ~2% of corpus** |
| abc | `data/` + `episode.mcap` `dir_depth=3` | 130,703 | train 129,032 + val 1,671 | 3,591 | no |
| agibot | per-base `proprio_stats` `dir_depth=2` | 164,493 | alpha 35,651 + beta 128,842 | 2,697 | no (this dump) |
| egoverse | `*.zarr` `max_depth=2` | 2,224 | 2,226 zarr at dump root | 93 | 2 empty zarr only |
| hy_lance | immediate `table_*` | 250,304 | `tables.json` 250,304 / 22 tables | 2,162 | no |
| droid | LeRobot discover depth 5 | 95,600 | `info.json` 95,600 | 511 | no |
| galaxea | same | 20,662 | 227 repos, sum 20,662 | 488 | no |
| hifi_umi | same | 482,060 | 398 parts + meta 482,060 | 2,137 | no |
| kai0 | same | 23,963 | 7 repos, sum 23,963 | 213 | no |
| libero | same | 1,693 | `info.json` 1,693 | 7.6 | no |
| rmbench | same | 600 | `info.json` 600 (`rmbench_all` same) | 12 | no |
| robotwin | LeRobot bind | — | no `datasets/robotwin`; `/mnt/open_source_data/RoboTwin` is h5/`.pt`, 0 `info.json` | — | not linked |

Hours = `n_steps / fps / 3600` from the scan manifest.

### das_gripper (unchanged)

Dump: `datasets/das_gripper` → `/mnt/open_source_data/DASGripper_slim`.

`dir_depth=3` only hits `Task/00001/01706/episode.hdf5` (Clutter Tidy-Up [Stage2]). Other tasks add a directory; `[STAGE 3]` goes deeper (up to 7).

| task | meta eps | scanned? |
|---|---|---|
| Clutter Tidy-Up [Stage2] | 12,985 | yes |
| Cooking_and_Kitchen_Clean | 1,898 | no |
| Folding_Clothes_and_Zipper_Operations | 36,437 | no |
| Organize_Clutter | 26,676 | no |
| Shoes_Handling | 724 | no |
| [STAGE 3] | 589,084 | no |
| **sum** | **667,804** | **12,985** |

12,985 × mean ~53.3 s ≈ 192 h. 667,804 × same mean ≈ **9,879 h**.

## 2026-09-09 — fix

`das_gripper.scan` reads only `{dump}/{task}/das_gripper_slim_meta.json` (or a single task folder that has one). No `dir_depth=3`, no walk of `[STAGE 3]`. Nested metas (e.g. under `00001/`) are ignored. No task meta → full `episode.hdf5` walk (unit tests). Do not bump global `SCANNER_VERSION` (other dumps are already correct).

`--rescan` 2026-09-09: **667,804** episodes, **997,666,055** steps @ 30 fps → **9,247 h**, 24.4 min → `{dump}/.cache/episodes`. Matches the 6 task metas. Clutter mmap (12,985×2 wrists) is still valid; the other ~655k episodes have no JPEG mmap yet.

### Why the others are not the same bug

- **abc**: layout is `data/{train,val}/<task>/episode_*/episode.mcap` (3 dirs under `data/`). Train+val reports sum to cache. `abc_train_meta.json` is train-only (129,032).
- **agibot**: layout is `{alpha,beta}/proprio_stats/<task>/<ep>/proprio_stats.h5`. Meta task sets == disk `proprio_stats` task dirs. Beta has 26 observation-only tasks with no proprio; they are not in the meta episode list. This copy is 164k episodes / ~2.7 kh, not the public “1M+ trajectories” AgiBotWorld figure — that is dump size, not a depth miss.
- **egoverse**: all 2,226 `*.zarr` sit at `/EgoVerse/aria` root. Dropped: `2025-11-16-04-59-56-244000.zarr` (no `zarr.json`, no pose arrays) and `2025-11-22-23-48-44-123000.zarr` (`total_frames=0`). Parent only contains `aria`.
- **hy_lance**: 22/22 `table_*`; episode count matches `tables.json`. Frames 233,467,612 vs official 233,600,314 (−132,702, ~0.06%).
- **LeRobot**: repo nest depth is 0–2 (`hifi` `chunk-*/part-*`, `kai0` `Task_*/…`). `max_depth=5` is enough. Cache `n_records`/`n_steps` equal summed `info.json` `total_episodes`/`total_frames`. Galaxea extra top dir is sidecar `meta/`, not a repo. Parent `lerobot_info.json` (110 eps) is a leftover demo, not this dump.

### Mix / provisioning (not dir_depth)

- `robotwin` is in `ALL` but `datasets/robotwin` is missing. On-disk `/mnt/open_source_data/RoboTwin` is Official h5 + self_collect `.pt` flow, not LeRobot; current `robotwin.py` → `scan_lerobot` would index 0.
- `rmbench_all` exists; `info.json` is the same 600 / 416,711 as `rmbench`.
