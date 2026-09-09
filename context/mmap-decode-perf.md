# Mmap decode opts 1/2/3 — perf

## 2026-09-07 — Build-time estimate (item 1 off)

Cam-frames = `n_steps × n_cams` from scan index. 1-proc fps = this morning native decode + pad + JPEG (hot). Wall clock uses **8 effective NFS workers** (16 started, ~50% busy).

| dump | cam-frames | 1-proc fps | 1-proc | ~8 workers | mmap now |
|---|---|---|---|---|---|
| hifi_umi | 577M | 190 | ~35 d | **~4–5 d** | no |
| droid | 83M | 301 | ~3.2 d | ~10 h | no |
| kai0 | 69M | 293 | ~2.7 d | ~8 h | no |
| das_gripper | 62M | ~117 | ~6.2 d | ~19 h | no |
| egoverse | 30M | 435 | ~19 h | ~2.5 h | no |
| libero | 0.55M | — | — | reused | yes |
| rmbench | 1.25M | — | — | reused | yes |

abc / agibot / galaxea / hy_lance: no scan index yet. robotwin missing on disk.

If NFS stays at the old ~44 fps hifi calibration, hifi is ~9–10 days. Optimistic hot 190 fps × 16 workers linear would be ~2 days; 8 effective is the middle number.

## 2026-09-07 — Dropped decode-time letterbox

Item 1 removed: MP4 decode is native; `write_frame_cache` still letterboxes. `letterbox_content_hw` stays only for pad.

## 2026-09-07 — Microbench on real dumps

Measured decode + letterbox-pad + JPEG q85. No cache write. 1 process.

| Opt | Case | Old | New | Note |
|---|---|---|---|---|
| 1 letterbox-at-decode | kai0 480×640 AV1, 400f | 293 fps | 249 fps | **slower** |
| 1 | droid 180×320 AV1, 167f | 301 fps | 297 fps | already small |
| 1 | hifi 512×640 H264, 400f | 190 fps | 162 fps | **slower** |
| 2 shared packed file | hifi 5 eps / 2121f (hot) | 155 fps | 164 fps | +6% |
| 3 stills per-frame | egoverse 400 JPEG | 130 fps (368MB stack) | 435 fps | **3.3×** |
| 3 | libero 214 PNG 256² | 445 fps | 508 fps | +14% |

Item 1: AV1/H264 still decode native res; `reformat` is extra work vs one `cv2.resize`.

Item 2: repo0 `head_main` is **1125 episodes in one 7.2GB file**. Per-episode span already avoids re-decoding the whole file; shared-once mainly saves 1125× open/seek on NFS.

Item 3: win is not packing source JPEG as-is (that would stretch). Per-frame imdecode→letterbox→encode avoids a `T×H×W` stack.

## 2026-09-09 — mmap train-path 500+500

`benchmarks/dataloader_throughput.py`, mmap on, batch=4, workers=16, history=0, shuffle=False, `--profile`. Warmup 500 + measure 500. `das_gripper` prebuild x32 was running on the same node (numbers are conservative).

| dump | cams | init | first batch | **samp/s** | parent p50 / p95 (ms) | steps |
|---|---|---|---|---|---|---|
| egoverse | 3 | 1.1s | 4.3s | **855** | 0.43 / 25 | 10.1M |
| kai0 | 3 | 2.2s | 1.2s | **535** | 0.46 / 42 | 23.0M |
| libero | 2 | 0.4s | <1s | **574** | 0.33 / 50 | 273k |
| rmbench | 3 | 0.3s | <1s | **529** | 0.43 / 49 | 417k |

Earlier 80+80 mmap (idle node): rmbench 1471, libero 707, egoverse 1112. Longer window + disk contention both pull these down.
