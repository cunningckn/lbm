# Scan index cache — design

## 2026-09-03

Cache `list[EpisodeRecord]` after the first full `scan_root`. Later loads skip tree walks.

On disk, only `{dataset_root}/.cache/episodes/` (no fallback directory, no env override, no silent rescan):

```
.cache/episodes/
  repos.jsonl         # interned LerobotDump.info by repo (written first)
  episodes.parquet    # one row per episode
  manifest.json       # commit last: spec, n_records, complete
```

Missing directory → scan and write. Incomplete / spec mismatch / missing endpoint files → `ScanIndexError`, pass `rescan=True`. Never write a truncated (`max_episodes`) index.
