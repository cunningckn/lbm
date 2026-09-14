# MolmoMotion-1M 全量 mmap 转存验收报告

记录日期：2026-09-15

分支：`feature/tc_covert_adapt`
生产成品：`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1`

## 结论

全量转存已完成。腾讯云 TI-ONE 重试任务
`train-1673540587627874048` 在单实例、100 CPU、0 GPU 配置下成功结束；成品根目录
和全部 8 个发布组件（7 个子集加 assets）均有 `READY.json`。发布根目录可独立迁移到
金山云；所有运行时索引均为相对路径，迁移后按本报告的 SHA-256 命令复验即可。

首次任务 `train-1673531756924119808` 在 MolmoSpaces 的上游零字节 3D NPZ 成员处失败。
修复 `e5256c8` 保留了该条样本的 2D 与相机数据，把无法取得的 3D 值表示为 NaN 和全 false
可见性，并在成品元数据中记录原始成员名；重试任务只重建未完成部分，复用三个已有 READY
子集。

## 发布内容

| 组件 | records | trajectory rows | 分片 | 已分配磁盘字节 |
| --- | ---: | ---: | ---: | ---: |
| DROID | 23,532 | 132,934,730 | 92 | 2,933,198,848 |
| EgoDex | 302,464 | 5,558,526,670 | 1,182 | 125,627,121,664 |
| HD-EPIC | 12,746 | 106,021,227 | 50 | 2,391,687,168 |
| MolmoSpaces | 140,701 | 3,682,273,450 | 550 | 83,691,479,040 |
| Xperience | 196,749 | 1,730,848,588 | 769 | 45,044,293,632 |
| YTVIS | 1,035 | 6,903,248 | 5 | 155,176,960 |
| Stereo4D（metadata/index-only） | 23,011 | — | — | 26,230,784 |
| 便携 MP4/H5 assets | 148,936 索引资产 | — | 8 archive | 78,795,771,904 |

共发布 700,238 条样本、11,217,507,913 条数值 trajectory rows、21,006 个正式组件文件。
Stereo4D、Xperience 以及部分 RGB 的上游发行限制保持原样：不伪造未发布的数值轨迹、相机或
RGB；详见各组件 `dataset.json` 的 limitations。

MolmoSpaces 的 `dataset.json` 含唯一 source anomaly：
`molmospaces/object/pick_place_2cam_randomized__house_1367__00000002__exo_camera_1` 的
`tracks/..._3d.npz` 为 0 字节。它仍是完整的一条发布样本，3D 窗口的 NaN/false 表示
`unavailable-empty-source-member`，不是有效的几何坐标。

## 完整性验证

1. 源 snapshot 固定为 Hugging Face revision
   `c3dec07d796ddeccdc8f5a35bf4920b3ee044feb`，82 个文件、285,601,107,377 bytes。
2. 生产任务 preflight 对 50 个 LFS 大文件执行 SHA-256 并通过。
3. `verify-release` 的 component manifest 验证通过。
4. `verify-release --verify-files` 对 8 个组件清单中的全部 20,990 个文件重新 SHA-256，全部通过。

机器可读结果：

- `reports/molmo_motion_full_release_manifest_verify_20260915.json`
- `reports/molmo_motion_full_release_file_verify_20260915.json`

迁移到金山云后执行：

```bash
cd /home/tione/workspace/kainingchen/lbm
PYTHONPATH=tools/molmo_motion_cache/src python3 -m molmo_motion_cache verify-release \
  --output /path/to/molmo-motion-1m-mmap/v1 --verify-files
```

## 磁盘占用

| 范围 | 已分配字节 | 表观字节 | 文件数 |
| --- | ---: | ---: | ---: |
| 原始 snapshot | 294,007,590,912 | 294,006,011,802 | 239 |
| 正式发布组件（7 subsets + assets） | 338,664,960,000 | 338,615,052,683 | 21,006 |

独立发布包比原始 snapshot 的已分配大小增加约 15.19%。assets 在腾讯云 CFS 与原始 MP4/H5
使用 hardlink，因此原始数据仍在场时，原始加正式发布组件去重后的占用为
553,878,913,024 bytes，**增量**为 259,871,322,112 bytes。复制到金山云时应以发布目录
自身的 338,664,960,000-byte 独立容量规划（跨文件系统传输会物化这些 hardlink 数据）。

`v1` 根目录的瞬时总占用为 422,335,279,104 bytes，不应当作为成品尺寸：其中包括 jobs、pilots
和 83,538,362,368-byte 的首次失败 MolmoSpaces partial。该 partial 已移到
`_jobs/kaining-molmo-motion-mmap-260915-004849/failed_molmospaces_partial-rbi0xmw0`，保留审计、
排除发布路径。

## 数据读取基准

所有项目采用同一组参数：128 次计时请求、16 次预热、8 帧 × 32 点随机窗口、固定 seed
`20260915`。先以相同的原始 source 候选建立索引和 warm cache，再分别计时 tar/NPZ 解码和
Parquet 索引后的 mmap NPY 窗口读取；每组 checksum 都相等。

| 子集 | 原始 samples/s | mmap samples/s | 吞吐提升 |
| --- | ---: | ---: | ---: |
| DROID | 210.91 | 5,687.47 | 26.97× |
| EgoDex | 169.17 | 1,961.11 | 11.59× |
| HD-EPIC | 241.59 | 6,041.76 | 25.01× |
| MolmoSpaces | 191.01 | 5,468.42 | 28.63× |
| Xperience | 359.78 | 4,734.60 | 13.16× |
| YTVIS | 287.41 | 6,183.31 | 21.51× |

该测试测量的是 warm-cache、单进程数值窗口加载；不包含 RGB/video decode、PyTorch 多 worker
调度、GPU 传输或冷盘 I/O。因此它证明转存消除了每样本的 tar 寻址与 NPZ 解码开销，但不应
被解释为端到端训练吞吐的保证。

每个子集的机器可读结果分别在：

- `reports/molmo_motion_full_benchmark_droid_20260915.json`
- `reports/molmo_motion_full_benchmark_egodex_20260915.json`
- `reports/molmo_motion_full_benchmark_hdepic_20260915.json`
- `reports/molmo_motion_full_benchmark_molmospaces_20260915.json`
- `reports/molmo_motion_full_benchmark_xperience_20260915.json`
- `reports/molmo_motion_full_benchmark_ytvis_20260915.json`

## 金山云迁移

从腾讯云复制正式成品时只传 `v1` 的发布内容；不要把 `_jobs`、`pilots` 或首次失败的审计 partial
当作训练数据同步。保留 hardlink 关系可使用：

```bash
rsync -aH --info=progress2 \
  --exclude '_jobs/' --exclude 'pilots/' \
  /home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1/ \
  <kingsoft-host>:/target/Datasets/molmo-motion-1m-mmap/v1/
```

完成后在金山云运行前述 `verify-release --verify-files`。训练读取只依赖 `subsets/`、`assets/`
和其 Parquet/NPY/JSON 索引，不依赖源 tar 或绝对路径。
