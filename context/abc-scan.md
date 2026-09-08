# ABC scan

## 2026-09-03 — walk `episode.mcap`, ignore json

Listing is `os.walk` via `iter_named(..., "episode.mcap")` starting at `{dump}/data` (not the dump root). `abc_train_meta.json` is not used. Language comes from `path.parent.parent.name` (`data/train/<task>/episode_*/episode.mcap`).

Do not walk dump-root `.cache` (HuggingFace download mirror, ~300k dirs) or `meta/`. `iter_named` prunes any `.*` directory. Walking the ABC root used to sit at `scan abc: 0ep` for hours inside `.cache/huggingface/download/...`.

`n_frames` is **not** MCAP footer duration. Footer `statistics.message_start/end_time` includes cameras (~30 Hz) whose span is wider than state/action (~287 Hz). `read_vectors` resamples the **overlap of the 8 scalar topics**:

```
t0 = max(first log_time of each state/action topic)
t1 = min(last  log_time of each state/action topic)
ticks = np.arange(t0, t1 + 1, round(1e9 / fps))
n_frames = len(ticks)
```

Scan uses the same `_resample_ticks` as `read_vectors`. `t0`/`t1` come from MessageIndex on the **first / last / max-end-time chunks** that contain each scalar channel (one sequential read per those chunks), not every chunk and not footer span.

Index-block read must start at `min(all message_index_offsets)`, not the first scalar offset. Camera indexes are often written first; using scalar-min + full `message_index_length` over-reads and raises `unpack requires a buffer of 4 bytes` (this aborted `build_scan_index` and made the tqdm line look empty).

ABC files are written by `python mcap-protobuf-support 0.5.3; mcap 1.3.0` (append-in-write-order). On 9 episodes across the dump (idx 0 / 100 / 1k / 12k / 43k / 50k / 86k / 100k / last): chunk `message_start_time` is monotonic in file order, adjacent chunks do not overlap, per-chunk MessageIndex timestamps are monotonic, each scalar channel's true min/max live in the first/last chunk that contains it (`first_miss=last_miss=0`). Not a spec guarantee for arbitrary MCAP; it is how this dump was recorded. 458 MB file 3.2s → 0.32s vs full-index.

This is an empirical match on ABC-130K files, not a spec proof. Checked 5 episodes (first / 1/3 / 2/3 / last / idx 12345): per-topic **first, last, and message count** from MessageIndex equal fully decoded `log_time`s (`dt0=dt1=0`), index counts equal `statistics.channel_message_counts`, no duplicate channel IDs, no chunks missing indexes, `log_time==publish_time`. So `n_frames` matching is not a 100 ms tick coincidence. Footer `arange` was off by 1 on 4/5 of those files.
