# MolmoMotion JPEG224 实施与验收报告

日期：2026-09-15；分支：`codex/molmo-motion-completion`；实施前 HEAD：`56541c9`。
代码提交：`1924a7c`（`feat(data): add verified JPEG224 pilot and portable joint reader`）。
依据：`MOLMO_MOTION_JPEG224_NEXT_PLAN_20260915.md`。

## 结论与发布状态

已实现并实测 JPEG224 转存、跨视频分片、轨迹/相机联合读取、100 段扩样、内容区画质/容量评估、0/1/4 worker 三轮成对性能测试，以及含 RGB 的金山云独立迁移验收。

**交付是可复核的实验包，不是正式同步训练 release。** 源二维坐标的像素中心/边界约定尚缺生成端证据；标注时间与编码时间仍未统一验证。两项状态均保持 false，未覆盖旧 v1，未启动全量 RGB 转存，未删除 MP4。源投影吻合不能替代上述证据。

## 已交付代码

- `geometry224.py`：与 SHA 固定的 PI0 数据预处理函数对齐；返回 224 RGB、实际内容尺寸、padding、A；拒绝尺寸/类型不符。
- `rgb224.py`：tar offset 流式解码、只存选定 JPEG、每组独占 writer、2 GiB 目标分片、跨 shard 索引、完整校验后发布 PILOT_READY；完成品严格按指纹复用。失败 staging 保留，未实现自动恢复未完成组。
- `generic_reader.py`：复用 worker reader，返回原始及 224 轨迹/K、按源索引选择相机、内容 mask、有效监督 mask、frame/point IDs、两种时钟；JPEG 默认读取。
- `evaluate_rgb224.py`、`benchmark_rgb224.py`：独立画质/空间检查和联合 reader 性能测试。
- `export_rgb224_pilot.py`、`verify_rgb224_migration.py`：四视频紧凑联合包与禁止源路径访问的跨云验收。
- `requirements-rgb224.txt`、新增 6 项测试、`run_acceptance_tests.py`：依赖锁定与可重复验收。

代码在 `tools/molmo_motion_cache/` 下，运行命令见该目录 README。运行时不导入 PI0 训练框架。源构建版本进入成品指纹；格式化前的实际构建源代码保存在 `JPEG224_BUILD_SOURCE.tar.gz`。最终版本另增加了 reader 初始化时的 NPY 索引哈希检查，19 项测试及金山云读取使用最终 reader；不能用最终代码的不同指纹静默续跑旧实验目录。

## A/B：像素、坐标、相机、时间

参考服务器文件 SHA-256 复核为 `85c66cd60814b8169c85a2fae52026cfd80f89942fd8bb1ed3b0b08f594a5396`。通过 AST 提取原函数，避免导入 torchvision 或模型代码。

1. 7 类尺寸（含 641×479、224×224）、彩色梯度及棋盘格，编码前与冻结函数逐像素相等；实际内容尺寸/padding 检查通过。641×479 得到 224×167、padding=0/28/0/29。
2. 100 段真实视频首/中/尾共 300 帧：编码前与服务器参考函数逐像素相等；PNG 内存往返无损；300 帧缓存解码与同参数重新 JPEG 编码后解码相等。PNG 文件仅在报告目录保留 12 张，不写入缓存。
3. 显式采用 **未验证的 integer-center 假设**：`x224=(x+.5)*sx-.5+left`；`K224=A@Ksource`。实际整数内容尺寸决定 sx/sy，不使用单一理想缩放比例。单元测试包括投影代数一致性与 NaN。
4. 100 段首/中/尾联合选择通过；原 3D、visibility 不变。稀疏相机测试使用非连续且乱序的源 frame indices，缺失帧拒绝，不按窗口数组位置误取。MolmoSpaces K 是像素域静态矩阵，其他动态/归一化情况未冒充支持。
5. 使用源 3D、c2w 的逆矩阵与 K 投影，100 段抽检的每视频中位残差为约 `4.77e-6～4.60e-5` 原图像素。这支持像素单位及相机解释，**无法区分整个源坐标系整体偏移半像素的约定**，因此未将源约定标为 verified。
6. 不重采样、截断、丢帧或放宽时差阈值。保留 frame_id、原始 PTS/time_base、`frame_id/annotation_fps`。时间语义仍为 false，只允许显式 frame-index 实验。

证据：`JPEG224_TEST_RESULTS.json`（19/19，无 skip）、`JPEG224_QUALITY_GEOMETRY.json`（逐视频）、`JPEG224_SAMPLE_BUILD_CONTRACT.json`（样本及参数）。独立工具范围 Ruff、compileall 通过，不宣称全仓库 lint 通过。

## C：100 段扩样、质量、容量

固定种子 20260915；包含历史四视频，然后按动作配置/house/视角/长度分桶/训练测试 split 分层轮转抽样。共 95 个 house、4 类 pick-place 配置、100 个唯一视频、26,695 帧。样本列表完整保存在构建 contract 的 inputs 中。不是对整个 MolmoMotion 七子集的随机代表样本。

| 同一批视频 | 字节数 | 十进制 MB | 相对源 MP4 |
| --- | ---: | ---: | ---: |
| 原 MP4 载荷 | 35,867,293 | 35.87 | 1.00× |
| JPEG224 q85、4:2:0 | 170,373,267 | 170.37 | 4.75× |
| JPEG224 q95、4:2:0 | 280,395,567 | 280.40 | 7.82× |

q85 比 q95 少约 **39.24%**。q95 在相同 RGB224 上编码计量后丢弃，不存一份完整 q95 缓存。q85 完整缓存含索引/元数据/清单：逻辑 **171,871,091 B**，文件分配 **171,954,176 B**；不是只算载荷，也不是目录 du。

每帧 JPEG 字节（min / p50 / p95 / max）：q85 为 3480 / 6367 / 8173.6 / 9109；q95 为 5517 / 10525 / 13689.3 / 15392。

| 内容区质量，不计黑边 | q85 | q95 |
| --- | ---: | ---: |
| 每视频 PSNR 的均值，dB | 37.32 | 40.65 |
| 每视频 PSNR 范围，dB | 33.28–41.26 | 35.65–44.73 |
| 300 个首/中/尾内容区 SSIM 均值 | 0.9585 | 0.9794 |
| SSIM 最低值 | 0.8966 | 0.9394 |

PSNR 使用每视频全部帧内容区 MSE，再平均视频 PSNR；不是全像素池化 PSNR。SSIM 只计算 300 帧，不宣称所有帧 SSIM。`JPEG224_COMPARISON.png` 每行一个中间帧，列依次为无损 RGB224 / q95 / q85；夹爪轮廓可辨，细纹理有差异。尚未完成对象/夹爪 ROI 分层定量评估或训练效果验收，q85 仍是候选而非已认证无影响的默认质量。

源索引已知唯一视频 140,701、总帧 36,881,113。按扩样帧均字节粗估：q85 图像载荷 **235.38 GB**，q95 **387.39 GB**；40 B/帧索引另约 **1.48 GB**，还需 metadata/manifest、数值数据和空间余量。抽样未做总体权重校正，不能当精确容量承诺。

受控 4 worker 构建阶段 20.84 秒，约 1281 帧/秒；单 worker 峰值 RSS 最大 405,544 KiB（约396 MiB）。该计时从建立组任务前开始，包含组解码/编码/组校验，**不包含最初全局索引读取、源指纹扫描及最终根清单发布**；不是整条命令墙钟。线性外推约 8 小时，仅为同设备条件下解码/编码阶段量级，不是全量 SLA。验收含双质量重编码/SSIM/原始数值检查另用 117.39 秒。

## D：分片、mmap 与恢复

100 段按唯一 video/view 写入 4 个 writer 组，每组多个视频共用 bin；小样本没有达到 2 GiB 上限。四视频 smoke 将阈值调为 1 MiB，实际跨多个 bin 读取首尾帧通过；另有跨 shard 乱序/重复帧、LRU 上限、损坏与错误续跑参数拒绝测试。

NPY 索引 64 位 offset/length，记录原始 PTS；视频内 frame_id 连续，视频可跨 shard。只在完整组/根清单校验通过后原子 rename；不把失败 staging 发布为成品。原尺寸与 224、质量/依赖/代码不同均不能混用续跑。未完成组的自动断点恢复仍待实现。

**mmap 范围仅为数值数组和帧索引**；JPEG 载荷采用每 worker 持久 seek/read 句柄（LRU 上限16），未声称 JPEG 载荷 mmap 加速。JPEG 解码仍占 CPU，mmap 不压缩磁盘容量。像素处理内存按帧受限，metadata 随视频数增长，尚未验证全量多小时运行的内存/句柄稳定性。

## E：独立联合 reader 性能

同一固定12视频、48请求、每请求8连续帧/最多32点。每 worker 持久数值/RGB reader；MP4 使用持久容器，按关键帧 seek 后解码到窗口尾，再做同样 Pillow224。0/1/4 worker、每组3轮、顺序交替。未清共享 page cache。

| worker | raw 稳态 req/s，中位数 | JPEG85 稳态 req/s，中位数 | raw 含启动 req/s，中位数 | JPEG85 含启动 req/s，中位数 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 14.99 | 171.43 | 4.24 | 5.69 |
| 1 | 14.64 | 139.77 | 4.08 | 5.17 |
| 4 | 30.41 | 63.60 | 3.58 | 3.79 |

原始每轮 p50/p95、RSS、初始化与请求 hashes 保存在 `JPEG224_BENCHMARK.json`。数值/K 的成对 hashes 相等，JPEG RGB 有损，不要求与 raw 相等。无损像素正确性基线见 A/B，而不是凭吞吐推断正确性。

重要限制：单进程初始化约8秒，4 worker各约11–12秒；复制读取完整数值元数据成本明显。4 worker并未比0 worker的短 JPEG 测试更快，且一轮 p95 达254.55 ms。稳态区间包括各 worker启动错位；没有用屏障强制统一开始。峰值 RSS 是进程生命周期 high-water mark，0 worker不同轮次共用进程，不能解释成各 codec 独占内存。两路径都读取共用 JPEG224 请求元数据来控制实验，非原训练框架初始化。

**这是独立联合 reader benchmark，不是 PyTorch DataLoader 或训练吞吐；没有 GPU 利用率结论。** 下一阶段应优化按样本过滤的数值索引加载，并接训练 DataLoader 后测长时间稳态。

## 含 RGB 的跨云验收

腾讯云紧凑包约9.7 MiB，4视频、1064帧，含重建的完整对应数值/相机组件、JPEG224/几何/两种时间信息，无绝对运行时路径。

- Tencent：`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/completion-tests/rgb224-joint-export-20260915`
- Kingsoft：`/mnt/kpfs/workspace/jinaoqun/Projects/lbm-completion-tests-20260915/rgb224-20260915`
- 两端 `python -I` 运行，完整文件哈希验证通过，audit hook 禁止原始 RAW/v1/RGB 根路径的 open，读取前后包内所有文件哈希不变。
- 两端逐样本 RGB、points2d/224、points3d、K/224、pose、PTS、frame_ids、mask 的 SHA-256 **完全相等**。金山云原腾讯云路径实际不存在。
- 金山云仅在先前测试 venv 内补 Pillow12.2.0 / PyAV17.0.1，没有改系统或训练环境；金山云 NumPy2.5.0 与腾讯云1.26.4 不同，读取结果仍一致。

证据：`JPEG224_MIGRATION_TC.json`、`JPEG224_MIGRATION_KS.json`。这项证据属于 JPEG224 联合包，不复用旧 DROID 数值迁移结论。全量迁移未执行，正式目标目录/配额仍须核实。

## 下一 agent 的明确剩余工作

1. 获取生成 NPZ 与渲染像素坐标约定的源码/版本证据，决定 integer-center 或 pixel-boundary。必要时只重建几何/联合层，不改写源数值；不能用当前极小投影残差直接认证半像素约定。
2. 获取发布版本的采样时钟证据，独立验证视频 frame-index 与 annotation-time 的语义，保持时差异常可观测；未通过前不发布同步 READY。
3. 对小目标/夹爪 ROI 扩充画质验收，制定可接受阈值并接训练验证，才选定生产 JPEG 质量。
4. 完善未完成组的安全恢复、全量元数据内存约束和长跑稳定性；优化数值索引初始化。随后接实际 DataLoader 长窗测试，不沿用当前短 reader 数字。
5. 上述门槛通过后，确定独立正式版本和足够空间，再执行全量 MolmoSpaces RGB 包及迁移；其他六子集缺失 RGB、Xperience camera、Stereo4D 数值仍按旧报告受阻，不能宣称七子集全模态完成。

新100视频实验目录：`tc_dev:/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/completion-tests/rgb224-pi0-20260915-100-q85`。报告均放成品之外，旧 v1/原尺寸 q85/q95/smoke 及失败现场均保留。
