# MolmoMotion v2 全量转存交付报告

日期：2026-09-16。构建代码：`56526e9`，远端分支 `feature/tc_covert_adapt`。全量构建于北京时间 10:20:34 完成；独立文件复核、清单审计、读取抽检及六项数值性能测试全部通过，验收任务退出码为 0。已交付当前源快照可用的全量 v2 组件，不代表所有上游模态或真实训练验收完成。

## 产物与范围

服务器：`tc_dev`。成品目录：`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v2`。原始数据及 v1 未覆盖。

本次全量指当前发布源快照内可用的数据，而非补齐所有上游模态。NPZ 数值解压、按字段合并为 NPY 分片，使用 mmap 切片读取；JSON 信息整理到 Parquet 索引与元数据，避免逐样本反复打开和解析小文件。视频和 H5 保留为可迁移资产归档及相对路径索引，不把视频重编码为 RGB224。

| 组件 | 记录数 | 逻辑大小 GiB | 说明 |
| --- | ---: | ---: | --- |
| DROID | 23,532 | 2.729 | 数值轨迹 |
| EgoDex | 302,464 | 116.967 | 数值轨迹 |
| HD-EPIC | 12,746 | 2.226 | 数值轨迹 |
| MolmoSpaces | 140,701 | 77.930 | 数值、相机信息，与 assets 校验来源配对 |
| Stereo4D | 23,011 | 0.024 | 仅元数据，不含重建轨迹/RGB |
| Xperience | 196,749 | 41.937 | 数值、权重及原始点身份 |
| YTVIS | 1,035 | 0.144 | 数值轨迹 |
| assets | 148,936 条索引 | 73.384 | 8 个归档及配套文件 |

七个子集合计 700,238 条索引记录，不等于独立视频数或训练窗口数。八个正式组件共 **338,594,913,604 bytes（315.341 GiB）**，不含根目录少量标记/日志。源清单的 82 个文件共 **285,601,107,377 bytes（265.987 GiB）**；成品组件逻辑大小为源文件的 **1.1856 倍（增加约 18.56%）**。解压后的数组以磁盘空间换取免解压、按需切片读取，并不以压缩率为目标。

大小口径：GiB = 2^30 bytes。以上按文件逻辑大小计算，适合估算完整复制的数据量，不是传输协议实际流量或本机新增物理占用。assets 可能与源文件共享硬链接，不能将多目录一次 `du` 的去重结果当作独立组件大小。失败首次构建留下的 `subsets/.droid.partial-rv99_9cb` 不计入成品，保留未删除。

## 完整性与可复现性

- 源 revision：`c3dec07d796ddeccdc8f5a35bf4920b3ee044feb`。
- 源 snapshot fingerprint：`45f5ac11cd78857e33945a5e22118be5e5faa8fb095dc46dc26b4fbf6a11a730`。
- 源清单检查通过：无缺失或大小不符；50 个 LFS 文件 SHA256 全部通过。
- 顶层及全部八个组件均有完成标记。独立清单审计通过：SHA256SUMS 无重复项、覆盖全部应覆盖的组件文件。
- 六个数值组件各读取首/中/尾样本成功；MolmoSpaces 与 assets 的版本/指纹配对通过。这是读取抽检，不是全部样本语义验证。
- 六个数值子集各 32 条源数据对照通过，共 192 条；各自 `verification.json` 均确认运行时路径为相对路径。这不等于全量逐元素源数据比对。
- 当前远端代码重跑工具测试：`16 passed in 1.20s`；重启前真实 DROID 两记录构建、发布和读取小测通过。
- 全文件哈希独立复核通过：八个组件共 20,990 个清单文件，全部为 `files-verified`，严格 source identity 检查通过。
- 六项性能测试均完成，源数据和成品消费字段的校验和全部一致，验收任务输出 `DELIVERY_VALIDATION_PASSED`，`exit_code=0`。

机器可读证据保存在 `reports/v2-evidence-20260916/`；云端验收目录为 `/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v2-validation`。

## 读取性能

固定 256 次请求、32 次预热、8 帧、32 点、随机种子 20260916；generic 子集每种轨迹选 8 个源记录，DROID 从全量索引随机选择。原始 tar/NPZ 与 mmap 使用相同请求并比较消费字段校验和。

| 子集 | 源读取 窗口/s | mmap 窗口/s | 倍率 | p50 ms 源→mmap | p95 ms 源→mmap |
| --- | ---: | ---: | ---: | ---: | ---: |
| DROID | 202.84 | 8,306.88 | 40.95× | 4.720→0.118 | 6.795→0.133 |
| EgoDex | 141.97 | 4,708.41 | 33.16× | 6.159→0.208 | 12.637→0.235 |
| HD-EPIC | 174.95 | 5,375.94 | 30.73× | 5.549→0.183 | 6.613→0.202 |
| MolmoSpaces | 159.39 | 4,931.20 | 30.94× | 6.067→0.201 | 6.764→0.217 |
| Xperience | 387.13 | 2,662.81 | 6.88× | 2.460→0.282 | 4.413→1.187 |
| YTVIS | 233.86 | 4,800.68 | 20.53× | 4.039→0.174 | 4.972→0.207 |

在这批固定小样本请求上，数值窗口读取为 6.88–40.95× 加速；不做未经测量的全数据集加权倍率推断。Stereo4D 只有元数据，assets 是视频/H5 归档，不套用数值窗口测速。

这是预热后的单进程数值窗口 microbenchmark，初始化/建索引不计时，不测 RGB 解码、真实 PyTorch DataLoader、多 worker 或 GPU 传输。因此不能把结果称为端到端训练加速。共享机器不清系统缓存，文件哈希扫描亦影响页缓存；不是冷盘测试。启动耗时、RSS/句柄、0/1/4 worker 稳态和真实训练吞吐本次未测，不据此宣称已满足这些后续验收项。

## 迁移到金山云

成品运行时索引使用相对路径，来源路径仅作溯源记录。复制整个 release 的正式组件和顶层元数据，不需要复制原始 NPZ/JSON 才能读取数值。先传完整小组件，在金山云核对 SHA256 并随机读取，再传完整成品。本次未启动跨云传输，尚无实际金山云验收结果。

从能访问 tc_dev 的金山云环境执行，按实际目的目录调整：

```bash
rsync -a --partial --info=progress2 \
  --exclude='.*.partial-*' --exclude='.build.lock' --exclude='build*.log' \
  tc_dev:/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v2/ \
  /YOUR/KINGSOFT/DATASET/molmo-motion-1m-mmap/v2/
molmo-motion-cache verify-release \
  --output /YOUR/KINGSOFT/DATASET/molmo-motion-1m-mmap/v2 \
  --verify-files --require-source-identity
```

目的端安装同一代码版本的工具及其读取依赖；校验通过前不将复制中的目录用于训练。命令不使用 `--delete`，不清理目的端数据。金山云无法解析 `tc_dev` 时，替换为可达 SSH 地址或通过中转机传输。

## 问题修复与限制

首次全量构建在 DROID 发布前校验中失败：公共 reader 要求 READY，但 staged 目录尚未发布。`56526e9` 增加严格限定隐藏 partial 目录的内部校验入口，公共 reader 仍拒绝未完成产物；随后重新构建，未伪造标记或降低对外读取门槛。

现有 release 没有 `frames.npy`/`frames-*.bin` 的 RGB224 物化数据。图像 mmap reader 的合成测试不等于本次数据含 RGB224，也不等于图像与轨迹时间/坐标已对齐。Stereo4D 仅元数据；其他上游受限或未发布模态未重建。真实训练 DataLoader、RGB 长时多 worker 测试及金山云小包验收仍属后续工作。
