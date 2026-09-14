# MolmoMotion-1M 全量转存工作记录

记录日期：2026-09-15

工作分支：`feature/tc_covert_adapt`

状态：实施中；本文是持续追加的工程记录，最终结果以同目录全量报告和成品根目录的 `READY.json` 为准。

## 1. 目标与固定路径

- 腾讯云原始数据：`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m`
- 腾讯云成品：`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1`
- 转存代码：`/home/tione/workspace/kainingchen/lbm/tools/molmo_motion_cache`
- 云任务资源：100 CPU、0 GPU、625 GiB memory、100 conversion workers
- 云任务日志：`<成品>/_jobs/<job-name>/build.log`

最终验收包括：完整源清单、七个发布子集、MolmoSpaces 已发布视频/H5 资产、逐组件 SHA-256、转存前后容量、相同请求的读取吞吐与校验和，以及可供其他 agent 复现的命令和踩坑记录。

## 2. 源数据补齐记录

Hugging Face local-dir 清单固定到 revision
`c3dec07d796ddeccdc8f5a35bf4920b3ee044feb`，共 82 个文件、
285,601,107,377 bytes。首次清点已有 56 个文件、252,551,294,298 bytes，
缺少 26 个文件、约 33.05 GB；金山云源目录缺少同样文件，不能从旧副本补齐。

腾讯云直连 `huggingface.co` 超时。可用下载链路为：

```bash
export HTTP_PROXY=http://10.0.0.222:8888
export HTTPS_PROXY=http://10.0.0.222:8888
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=1200
hf download allenai/molmo-motion-1m --repo-type dataset \
  --revision c3dec07d796ddeccdc8f5a35bf4920b3ee044feb \
  --local-dir /home/tione/workspace/kainingchen/Datasets/molmo-motion-1m \
  --include 'stereo4d/**' 'xperience/**' 'ytvis/**'
```

下载器使用 local-dir `.incomplete` 文件，连接中断后按字节断点续传。一次并发下载在
Xperience `tracks-0000.tar` 约 3.65 GB 处发生 `IncompleteRead`；随后缩小为单文件、
单 worker 重试并从 `3649044480/10818508800` 继续，而不是删除临时文件重下。
最终 82/82 文件的实际大小与清单完全一致，`present_bytes=expected_bytes=285601107377`。

## 3. 实施内容

1. 以官方 split JSON 为样本真值，不靠枚举 tar 成员决定数据集范围。
2. 对未压缩 tar 建立一次 byte-offset 索引，直接读取成员，不解包出百万级小文件。
3. DROID、EgoDex、HD-EPIC、MolmoSpaces、Xperience、YTVIS 轨迹统一为
   `(T,K,D)`；分片内展平写入 NPY，Parquet 保存 sample/object、offset、T、K。
4. 相机按动态 pose/intrinsics 或静态 K 分开连续存储；保留坐标约定和缺失原因。
5. Xperience 额外保留 `trust_weights` 与 `keep_mask`，单双手角色也做一致性检查。
6. Stereo4D 按官方发布范围转存 metadata/track index；源包未发布的数值轨迹、相机、
   RGB 明确标记为需上游重建，不伪造。
7. MolmoSpaces 发布包自带的 MP4/H5 tar 以 hardlink（跨文件系统时 copy）纳入成品，
   同时生成资产偏移索引。跨云 rsync 会正常传输实际字节。
8. 每个组件先写同文件系统 staging，抽样做源值一致性校验，通过后原子 rename 并写
   `READY.json`。完整发布根目录只有七子集均完成后才写顶层 `READY.json`。
9. 生产源文件校验 LFS SHA-256；成品每个组件生成 `SHA256SUMS`，支持迁移后全量复验。
10. 完整任务按已完成组件恢复；输出根有 `flock`，避免两个提交并发写同一目录。

## 4. 已完成验证

- 独立工具自动测试：6/6 通过，覆盖 DROID reader、tar offset NPZ、轴顺序、
  NumPy pickle 兼容、100 CPU/0 GPU 请求、完整发布组件校验。
- 真实数据 adapter 小样：EgoDex、HD-EPIC、MolmoSpaces、YTVIS 均完成构建，
  并通过源 NPZ/相机与成品 NPY 的逐字段抽检。
- NumPy 兼容：源中的对象数组由 NumPy 2.x 写入；腾讯云现有验证环境为 NumPy 1.26，
  读取层加入 `numpy._core` 兼容映射后真实 EgoDex hand 样本通过。
- 基准工具自检：EgoDex 小样 32 次相同窗口请求，原始 tar+NPZ 为
  161.34 samples/s，mmap 为 4,802.08 samples/s，校验和一致，约 29.76 倍。
  该数值只用于验证基准方法，不能替代全量成品的最终结果。
- TI-ONE 请求干跑：`Cpu=100000`、`Gpu=0`、`Memory=640000`、单实例，
  start command 使用 `WORKERS=100`、`SHARD_SIZE=256`，密钥不进入请求或代码快照。
- 七子集联合 smoke：DROID 1、EgoDex 4、HD-EPIC 2、MolmoSpaces 2、
  Xperience 4、YTVIS 2、Stereo4D 2 条记录全部完成，顶层状态为 `pilot-ready`。
- Xperience 额外字段基准自检：32 次相同请求原始 387.81 samples/s、mmap
  2,751.91 samples/s，校验和（含 `trust_weights`/`keep_mask`）一致，约 7.10 倍。

## 5. 踩坑与处理

| 问题 | 根因 | 处理 |
| --- | --- | --- |
| 腾讯云无法直连 Hugging Face | 出口连接超时 | 内网代理配合 `hf-mirror.com` |
| 大文件连接中断 | 代理长连接出现 `IncompleteRead` | 保留 `.incomplete`，单文件单 worker 断点续传 |
| 金山云副本也缺文件 | 上游旧副本并非完整 snapshot | 以固定 HF revision 的 82 文件清单为准补齐 |
| NumPy 1.x 无法读部分 pickled dict | NumPy 2.x pickle 引用 `numpy._core` | 读取层安装兼容 module alias，并用真实 hand NPZ 验证 |
| 初始 Xperience 字段不完整 | 仅转换轨迹/visibility 会丢置信度语义 | 增加 `trust_weights` NPY 和 `keep_mask` Parquet 列及 parity check |
| 记录数分片过小会制造大量成品文件 | 32 records/shard 对约 70 万 clip 不合适 | 生产改为 256 records/shard，smoke 仍用 2 |
| 300 CPU 规格容易遇到资源占满 | 专用资源组可调度余量有限 | 按最新要求降为 100 CPU/100 workers，并每 20 分钟监控 |
| 资产重复 hash 会额外读取数百 GB | MP4/H5 是已校验源文件的 hardlink | 生成清单时复用已通过 preflight 的 LFS SHA-256；迁移端仍可全量重算 |
| 首次 TI-ONE API 调用无响应 | tc_dev 直连 API endpoint 超时 | SDK `HttpProfile.proxy` 固定使用可达内网代理 |
| CPU-only 请求返回 `InvalidParameter [gpu]` | 沿用了 GPU 任务的 `GpuType=HCC-BW1000` | 按官方 CPU 示例改为 `Gpu=0`、`GpuType=""` |

## 6. 待完成验收

- [x] 82/82 源文件尺寸全部通过；LFS SHA-256 由生产任务 preflight 执行
- [x] 包含 Xperience 的七子集总 smoke 通过
- [ ] 提交并监控 100 CPU / 0 GPU TI-ONE 全量任务
- [ ] 顶层 `READY.json` 出现，七子集和 assets 均有组件 `READY.json`
- [ ] 全量成品 manifest 与抽样源值复验通过
- [ ] 统计同范围源/成品逻辑字节、实际磁盘占用、文件和分片数
- [ ] DROID 与五个通用数值子集完成相同请求的原始/mmap 读取基准
- [ ] 汇总最终测试报告与金山云迁移验收命令
