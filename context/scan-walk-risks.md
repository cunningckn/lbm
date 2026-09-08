# Scan walk risks (ABC-style `.cache` trap)

## 2026-09-03

ABC hung at `0ep` because `os.walk(dump)` sorts `.cache` first. `ABC-130K/.cache/huggingface/download/` mirrors the dataset (~300k dirs) and has no `episode.mcap`. Fix: `iter_named` prunes `.*`; ABC walks `{dump}/data` only.

Checked every on-disk dump under `datasets/` (robotwin missing):

| dump | how scan lists | dump-root trap dirs | same ABC risk? |
|---|---|---|---|
| abc | `iter_named` from `data/` | `.cache` 305k, `meta` 129k | fixed |
| das_gripper | `iter_named` from dump root | none now; `[STAGE 3]` nlink 2.3M is real data | no cache trap; still unbounded walk |
| agibot | `iter_files_at_depth(proprio_stats, .h5/.hdf5)` (skip `.*`, yield as found) | no dump-root `.cache`; used to `sorted(glob)` twice and hang ~3 min with no bar | fixed |
| egoverse | `glob("*.zarr")+glob("*/*.zarr")` then `sorted` | no `.cache`; ~100k zarrs at root | no; second glob lists inside every zarr before tqdm |
| hy_lance | `iterdir` `table_*` only | `.hfd` (tiny HF downloader meta) | no |
| kai0 | `discover_lerobot_roots` (skip `.*`, depth 5) | no `.cache`; `Task_*` then nested `info.json` | no |
| galaxea | same discover | no `.cache`; each task has `meta/info.json` | no |
| hifi_umi | same discover | `.hfd` skipped; repos at `chunk-*/part-*/` | no |
| libero / rmbench | discover finds `meta/info.json` at root and stops | `.mmap` 17–39k dirs, skipped because name starts with `.` | no (would be ABC-like if walk did not skip `.*`) |
| robotwin | no dump | — | — |

`discover_lerobot_roots` already skips `.*`. `iter_named` now does too. Do not walk dump root when a known data prefix exists (`data/`, `proprio_stats/`, `table_*`).
