# MolmoMotion-1M DROID mmap 转存试验报告

试验时间：2026-09-14
分支：feature/tc_covert_adapt
状态：通过；这是可迁移的 DROID 轨迹和相机标定试产物，不是全量 MolmoMotion-1M 成品。

## 交付位置

- 转存代码：/home/tione/workspace/kainingchen/lbm/tools/molmo_motion_cache
- 试产物：/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1/pilots/droid-pilot-512
- 本报告：/home/tione/workspace/kainingchen/lbm/reports/MOLMO_MOTION_DROID_PILOT_REPORT.md
- 原始输入快照：/home/tione/workspace/kainingchen/lbm/reports/molmo_motion_droid_source_snapshot.json
- 独立校验结果：/home/tione/workspace/kainingchen/lbm/reports/molmo_motion_droid_pilot_verify.json
- 速度基准原始结果：/home/tione/workspace/kainingchen/lbm/reports/molmo_motion_droid_pilot_benchmark.json

## 本次范围

数据总目录仍在传输，试验没有读取任何 .part、下载缓存中的 incomplete 文件或正在写入的文件。已完整可读的 DROID 输入为：

| 项目 | 数量 | 逻辑字节 |
| --- | ---: | ---: |
| 已完成 DROID tar 和官方注释 | 5 个文件 | 2,340,248,092 B（2.18 GiB） |
| 官方 DROID split | train 22,363 / test 1,169 | 23,532 clips |
| 本次试转存 | 512 条、均为 train | 769 个 motion ranges |

512 条记录来自完整的 DROID tracks-0000.tar 和 camera-0000.tar。DROID 的视频需要单独取得上游 DROID MP4 后重建；当前输入没有这些视频，因此本轮只验证轨迹和相机标定，不虚构 frames.bin 或视频帧。

## 转存实现

转换器使用官方 droid_split.json 做样本真值来源，而不是枚举临时解包目录。每条记录的 *_2d.npz 和 *_3d.npz 在 tar 中按成员读取，连续写入分片级 NPY：

| 源字段 | 试产物字段 | 处理方式 |
| --- | --- | --- |
| tracks_2d | points2d.npy | (T, N, 2) 展平后以 row_offset、T、N 重建 |
| points_3d | points3d.npy | (T, N, 3) 展平，float32 与 NaN 原样保留 |
| visibility / valid_3d | visibility2d.npy / visibility3d.npy | 单独的 bool 数组，不做全局补齐 |
| camera JSON | 三个标定 NPY | measured K、按 ds_dim 缩放的 K、外参分别存储 |
| caption、split、motion ranges、偏移 | Parquet | clips、tracks_index、cameras_index、objects 四张表 |

每个分片通过 NPY mmap 写入，读取器只加载 Parquet 索引一次，再按相对路径打开 NPY。运行时不需要原始 tar、NPZ 或 JSON，也没有绝对源路径、外部软链接或 pickle 依赖。

## 磁盘大小

下表先给出可比的 512 条逻辑输入闭包。它包括所选 NPZ tar 成员、去重后的 camera JSON 成员，以及写入 provenance 的三份官方注释 JSON（其中 droid_split.json 是构建索引的真值来源）；并非把 2.18 GiB 的完整 DROID tar 错当成 512 条样本的大小。

| 项目 | 逻辑字节 | 说明 |
| --- | ---: | --- |
| 所选 1,024 个 NPZ tar 成员 | 80,270,988 B | NPZ 内部仍是压缩容器 |
| 所选去重 camera JSON 成员 | 850,631 B | 相机文档按 scene 复用 |
| 三份官方注释 JSON | 17,099,292 B | split 真值和记录到 provenance 的 clips、video index |
| 512 条逻辑源输入闭包 | 98,220,911 B（93.67 MiB） | 可比的构建输入 |
| mmap 试产物 | 104,659,916 B（99.81 MiB） | 26 个常规文件 |
| 其中 NPY | 104,573,558 B | 热路径数值数据 |
| 其中 Parquet | 79,386 B | 索引和元数据 |

试产物相对该逻辑输入闭包为 1.066 倍，即增加约 6.6%。这是有意的空间换时间：NPZ 的压缩容器被解码成 mmap 可直接切片的连续数值数组。若只与所选 NPZ 和相机成员比较、而不把全局注释算入源闭包，输出约大 29.0%。因此该实验不声称“压缩节省磁盘”；它的目标是减少随机加载时的 tar 寻址、ZIP/NPZ 解压和大量 JSON 解析。

完整 DROID 容器的 2.18 GiB 与 512 条试产物不能直接计算压缩比，因为完整容器仍含 23,532 条 clip。全量转换完成后应重新报告同一范围的总源字节、总输出字节和分片数量。

## 正确性与可迁移性

- 单元测试：2/2 通过，覆盖轨迹展平/重建窗口与 DROID K 缩放。
- 构建期：随机抽检 24 条记录，源 NPZ/JSON 与写出的 NPY 数值逐字段一致。
- 独立复验：再抽检 64 条记录，2D、3D、两个可见性 mask、两种 K 和外参均一致；浮点 NaN 也按相等处理。
- 完整性：SHA256SUMS 覆盖 24 个交付文件，全部校验通过；清单哈希为 292b4b98c1750322f96a423deccfd8abd4d930777cfc314192ea746335fd139e。
- 迁移检查：试产物被复制至另一个根目录后，不访问原始数据即可读取 512 条索引和一个 (8, 32, 3) 轨迹窗口；校验和仍通过，二进制扫描未发现原始数据根目录字符串。该临时迁移副本已清理。

这里的 PILOT_READY.json 表示该试产物完成且可复制；它刻意不使用全量交付的 READY.json，以避免将未转的 subset 或视频错误标记为完成。

## Dataloader 读取速度

测试方法：同一随机序列，512 个样本，先 warmup 32 个样本；每次读取 8 帧 × 32 个点，并对相同字段做 NumPy materialization 和 checksum，确保计时不只是索引查找。随机种子为 20260914。

| 路径 | 512 次耗时 | 吞吐 | p50 延迟 | p95 延迟 |
| --- | ---: | ---: | ---: | ---: |
| 原始 tar + NPZ + JSON | 2.89298 s | 176.98 samples/s | 5.582 ms | 7.606 ms |
| Parquet 索引 + NPY mmap | 0.05563 s | 9,204.25 samples/s | 0.108 ms | 0.118 ms |
| 相对提升 | 52.01 倍 | 52.01 倍 | 约 51.9 倍更低 | 约 64.2 倍更低 |

结论只适用于本机的 warm-cache 随机轨迹窗口读取。它没有测量冷盘 I/O、网络文件系统争用、PyTorch worker 调度、GPU 传输或 DROID 视频帧解码；后四项应在视频到位后另行基准。这个速度结果的主要来源是跳过了 tar member 定位、每样本 NPZ 解压和重复 JSON 解析。

## 复现命令

    export PYTHONPATH=/home/tione/workspace/kainingchen/lbm/tools/molmo_motion_cache/src
    PY=/home/tione/workspace/kainingchen/xprobot_flow/.venv/bin/python
    RAW=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m
    OUT=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1/pilots/droid-pilot-512

    $PY -m molmo_motion_cache build-droid \
      --source-root "$RAW" --output "$OUT" --limit 512 --shard-size 256 --checks 24

    $PY -m molmo_motion_cache verify \
      --source-root "$RAW" --output "$OUT" --checks 64 --verify-hashes

    $PY -m molmo_motion_cache benchmark \
      --source-root "$RAW" --output "$OUT" \
      --samples 512 --warmup 32 --frames 8 --points 32 --seed 20260914

## 后续全量实施约束

1. 等待源 tar 以最终非 .part 名称落盘，再运行；转换器会天然忽略未完成文件。
2. DROID 全量可使用不带 limit 的 build-droid；现有试产物不应覆盖，应创建新的全量输出目录或先实施可验证的 resume 语义。
3. EgoDex、MolmoSpaces、HD-EPIC、YTVIS、Xperience、Stereo4D 需要各自适配器和真实样本校验，不能把 DROID 的字段假设套用到它们。
4. 视频到位后，为每个 clip 写入 frame payload 与 offset/length 索引，并把视频随机帧读取加入同一基准。
