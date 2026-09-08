# Dataloader throughput (--no-mmap)

## 2026-09-03 — batch=4, workers=16, warmup=80, max_batches=80, shuffle=False

Config: `--no-mmap --profile`, history=0s (current frame only), action=1s.

| dataset | cams | on-disk | init | first batch | samp/s | parent_wait p50/p95 (ms) | steps |
|---|---|---|---|---|---|---|---|
| libero | 2 | lerobot parquet images | 4.8s | 11.4s | **37.6** | 0.3 / 1004 | 273k |
| rmbench | 3 | lerobot mp4 | 3.1s | 77.6s | **20.5** | 0.4 / 1712 | 417k |
| egoverse | 3 | zarr jpeg | 70.0s | 4.9s | **51.8** | 0.4 / 635 | 10.1M |
| hifi_umi | 3 | 398× lerobot v3 mp4 | 398.2s | 29.9s | **4.4** | 0.4 / 5992 | 192M |
| kai0 | 3 | lerobot mp4 | 122.1s | 246.2s | **210.7** | 0.3 / 251 | 23.0M |
| hy_lance | 3 | lance | 7.2s | 20.5s | **103.9** | 0.4 / 325 | 233M |
| das_gripper | 3 | hdf5 + wrist mp4 | still scanning (~20min, 2048 hdf5) | — | — | — | — |

Remaining: das_gripper, galaxea, abc, agibot. robotwin has no dump.

Notes:
- `--dataset foo` follows the symlink; spec must be `--robot-type foo`.
- Default 8+20 batches only drains prefetch (16×4=64). Use warmup≥80 to measure past the queue.
- kai0 210 samp/s is sequential after a 246s worker-start; not random-access.
- Live mp4 path (`read_mp4_indices`) opens + seeks the file on every `__getitem__`.
- das_gripper scan: `os.walk` + `h5py.File` per `episode.hdf5`.
