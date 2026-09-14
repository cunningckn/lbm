# LBM / Molmo 数据转存开发日志

记录日期：2026-09-14。技术核对基线：`feature/tc_covert_adapt@d7d9149`。

本文整理本轮对话中的需求演进、腾讯云适配、MolmoMotion 转存设计、真实样本测试及 Git 治理。用户所说的“IBM”在本项目语境中指 **LBM（Large Behavior Cloning Model）**，下文统一使用 LBM。

目前交付的是 **DROID 子集 512 条轨迹与相机数据的可迁移 mmap 试产物**。在指定热缓存窗口读取测试中，吞吐由 **176.98 提升到 9,204.25 samples/s，约 52.01 倍**。完整 MolmoMotion-1M、多子集视频转换、金山云实际接收和训练端到端验证尚未完成。

## 1. LBM 优化数据吞吐的思路

### 1.1 LBM 是什么，数据瓶颈在哪里

LBM 是行为克隆训练项目，模型使用机器人观测学习动作。mmap 是该项目的数据读取机制之一，不是模型名称，也不是一种压缩算法。

机器人和运动数据常以 MP4、Parquet、NPZ、JSON 等形式发布。这些格式便于发布、压缩和分析，但训练会不断随机抽取短时间窗口：视频随机访问需要定位并解码，压缩 NPZ 往往要先解压数组，零散文件需要反复打开，元数据也可能被反复解析。若每个 epoch 重做这些工作，CPU 和存储会让训练等待数据。

LBM 的主要做法是把能复用的处理提前完成，让训练阶段直接按索引取数据。

| 层次 | LBM 中的处理 | 对读取的帮助 |
| --- | --- | --- |
| 数据扫描 | scan index 记录 episode、文件和必要元数据 | 减少反复遍历目录、识别结构的成本 |
| 数值缓存 | episode Parquet 数值列转成 `.npy`，以 `mmap_mode="r"` 打开 | 按需访问状态、动作、时间戳，减少反复解析整表 |
| 图像缓存 | 提前解码、缩放补边、编码 JPEG，写入 `frames.bin` 和 offset/length 索引 | 随机取帧无需每次从 MP4 定位解码；训练仍需 JPEG 解码 |
| 构建与读取分离 | 在训练开始前预构建，worker 读取已完成缓存 | 减少训练途中建缓存、争锁及重复工作 |
| 衍生量复用 | 按配置预计算 FK、归一化统计等 | 避免重复计算，并让统计口径与训练动作定义一致 |

现有 LBM README 的常见图像默认值是 224 尺寸补边、JPEG quality 85。这是机器人训练缓存配置，不能直接当成 MolmoMotion 几何轨迹缓存的默认值。

### 1.2 为什么 mmap 能快

mmap 将文件映射到进程的虚拟地址空间，读取数组切片时由操作系统按需调入对应页面。重复访问还能利用 page cache；连续数值布局也减少对象解包和文件级操作。

它不会让磁盘变成内存，也不保证所有读取都零拷贝。实际取点、构造连续数组和拷到 GPU 仍可能复制数据。文件系统延迟、访问局部性、图像解码和 worker 调度也会影响收益。本轮基准明确执行了数组 materialization 和 checksum，而不是仅计时创建 mmap 句柄。

工程上的核心是：**离线处理一次，数值连续存放，索引复用，按训练需要切窗口**。增加 worker 是配套手段，并发数应由任务实际资源与实测决定。

### 1.3 “转存”在本轮讨论中的具体含义

“转存”曾同时指跨云复制、数据目录整理和训练缓存预构建，后续需要区分三者：

| 操作 | 输入 → 输出 | 是否改变数据组织 |
| --- | --- | --- |
| 跨云搬运 | 腾讯云文件 → 金山云同一份文件 | 原则上字节不变，由校验和验收 |
| MolmoAct 接入 | 已有嵌套 LeRobot 仓库 → LBM 可发现的 dataset 目录 | 可能只是目录关联；当前 catalog 的 process 为 none |
| LBM 缓存预构建 | LeRobot Parquet/MP4 → 数值 NPY、JPEG 帧包及索引 | 改为适合随机训练读取的布局 |
| MolmoMotion 本次转存 | tar 内压缩 NPZ、相机 JSON、官方标注 → 分片 NPY、Parquet 和清单 | 保留运动语义，重排物理存储 |

因此，不能把 MolmoAct 现有脚本一概描述为“把任意原始格式转成 LeRobot”。对于已经是 LeRobot 的数据，真正服务于吞吐优化的是后续缓存预构建。

代码依据：[项目说明](../README.md)、[数据处理说明](../scripts/data/README.md)、[数值缓存](../src/lbm/dataloader/mmap/mmap_io.py)、[图像缓存](../src/lbm/dataloader/mmap/frame_mmap_io.py)。

## 2. 在腾讯云适配 Molmo 数据转存的工作

### 2.0 需求演进与腾讯云运行适配

对话先讨论在两个服务器间使用同一项目，再将金山云 LBM 项目复制到腾讯云开发；随后要求在 `feature/tc_covert_adapt` 隔离适配工作。目标从现有 MolmoAct 数据接入，扩展为 MolmoMotion-1M 的跨云可迁移缓存。用户启动原始数据传输后，要求先用到达的真实数据验证代码并报告容量与读取速度。

| 用途 | 主机 | 路径 |
| --- | --- | --- |
| 原始数据来源 | `mindon_server2` | `/mnt/open_source_data/molmo-motion-1m` |
| 金山云原项目 | `mindon_server2` | `/mnt/kpfs/workspace/jinaoqun/Projects/lbm` |
| 腾讯云开发项目 | `tc_dev` | `/home/tione/workspace/kainingchen/lbm` |
| 腾讯云输入 | `tc_dev` | `/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m` |
| 用户确认的成品根目录 | `tc_dev` | `/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1` |
| 已完成试产物 | `tc_dev` | `/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1/pilots/droid-pilot-512` |
| 金山云建议接收目录，尚未交付 | `mindon_server2` | `/mnt/kpfs/workspace/jinaoqun/Datasets/molmo-motion-1m-mmap/v1` |

这里记录的是项目与路径约定；本轮没有证据证明已经在应用中完成“项目永久绑定两台服务器”的配置。

已整理入分支的腾讯云适配包括：

- 数据目录与运行参数：`scripts/data/convert_50cpu.sh` 支持 `DATASET`、`SOURCE`、`DATA_BASE`、`LBM_DATASETS`、`PYTHON_BIN`、`UV_BIN`、`WORKERS`、`CHECK_ONLY`、`BUILD_MMAP`；`convert_job.sh` 复用同一入口。
- 资源控制：默认 mmap worker 为 `min(nproc, 48)`，拒绝大于检测可用 CPU 的配置；OMP/MKL 默认单线程。脚本名含 50cpu，但不会自动申请 50 CPU。调度任务仍需检查 cgroup 配额与实际吞吐。
- 数据接入：catalog 增加 InternData-A1、MolmoAct、RoboCOIN；对应 adapter 明确绝对关节动作定义并沿用归一化链路。MolmoAct adapter 为 Franka、10 Hz、7 维 state/action，三视角 `first_view`、`second_view`、`wrist_image`。
- 独立转换工具：`tools/molmo_motion_cache` 只依赖 NumPy 和 PyArrow，包声明 Python >=3.10；本次在既有 Python 3.12 环境运行。基础转换与 reader 不导入 LBM、Torch 或 CUDA。

历史检查中，tc_dev 交互环境曾只暴露 1 个可用 CPU；现有 Torch 环境导入曾因缺少 `libgalaxyhip.so.5` 失败，外部包源也曾超时。这促成了独立 CPU 工具的设计。Shell 语法、配置检查和独立工具测试通过，不等于完整 LBM 训练环境验证通过。

### 2.1 MolmoMotion-1M 数据转存思路

#### 2.1.1 数据任务决定缓存结构

MolmoAct 主要服务于机器人动作学习，本项目接入的是状态、动作、图像和任务文本。MolmoMotion-1M 服务于视觉运动理解/预测，涉及 2D/3D 点轨迹、对象或手部标注、有效性、相机、时间、caption 和运动区间。部分来源有机器人数据，但不能把完整运动语料统一当成关节动作数据集。

因此选用独立 motion cache：借鉴 LBM 的数值 mmap 和图像打包方法，保留运动语义及后续查询点、时间窗口的灵活性。LeRobot 不作为这一流程的必要中间格式。

#### 2.1.2 完整方案的设计要点

1. **从官方 annotations/split 建立样本清单。** 保留来源与划分，关联 clip、object、相机、标注视图，避免仅枚举 tracks 而遗漏数据。
2. **轨迹按字段归并。** 统一逻辑轴为 `(T, K, D)`，分片内展平拼接，索引保存 offset、T、K、dtype。变长记录不做全局 padding。
3. **保持数值与几何语义。** 保留精度、NaN、2D visibility 和 3D validity；分别记录坐标系、相机内外参、分辨率、FPS、帧索引。缩放图像时同步变换 2D 点和内参。
4. **图像按视频资产复用。** 计划预解码后以 JPEG/PNG 字节包和帧索引存储，避免同一视频因多个对象重复保存；不提前展开所有训练窗口。原生分辨率、JPEG q95 是待验证配置，PNG 用于无损对照。
5. **缓存可搬迁。** 运行时使用稳定 ID 与根目录内相对路径，源路径不参与身份；迁移后仅替换 data_root。源码和依赖版本随成品交付。
6. **按分片完成与验证。** 完整方案拟支持约 2–8 GiB 分片、有界内存、指纹匹配的 resume、SHA256 清单和最终完成标记。当前 pilot 尚未实现全部能力。

#### 2.1.3 NPZ、JSON 小文件如何统一处理

统一的是字段、身份、索引和存储布局，不是把所有内容塞进一种文件格式。

| 原始内容 | 当前 DROID 实现 | 训练读取方式 |
| --- | --- | --- |
| `*_2d.npz` 中轨迹 | `points2d.npy`，分片级 float32 数组 | 按行偏移恢复形状，再切窗口 |
| `*_3d.npz` 中轨迹 | `points3d.npy`，保留 float32 和 NaN | 同上 |
| `visibility`、`valid_3d` | `visibility2d.npy`、`visibility3d.npy` | 分别保留 bool mask |
| 相机 JSON | measured K、按 ds_dim 缩放的 K、外参，三个 NPY | 数值切片，不再读原始 JSON |
| caption、split、运动区间、来源引用 | `clips.parquet`、`objects.parquet` | 加载索引/元数据后复用 |
| 数值位置与相机位置 | `tracks_index.parquet`、`cameras_index.parquet` | sample ID → shard/offset |
| 来源文件信息 | `provenance/source_manifest.parquet` | 审计使用，常规读取不依赖源文件 |

例如 A 的形状为 `(10,4,3)`，B 为 `(8,3,3)`，可依次存成同一 `(64,3)` 数组。A 保存 `row_offset=0,T=10,K=4`；B 保存 `row_offset=40,T=8,K=3`。reader 不必把整条记录都读入内存才能定位窗口。

输入已经打成 tar 时直接读取成员，不先解压成成千上万个松散文件。当前输出也不是“一个 NPZ 对应一个 NPY”：512 条记录合并进 2 个分片，总计 26 个常规文件。原来 1,024 个 NPZ 是 tar 内成员数，不能把它当成输入文件系统上的 1,024 个独立 inode。

JSON 字符串并非在成品中完全消失：当前 `clips.parquet` 保留 `motion_ranges_json`，`objects.parquet` 另存可查询区间；数据集/分片仍有少量说明 JSON。数值窗口热路径不解析原始 NPZ 或相机 JSON。计划中的完整原始 annotations 归档尚未生成。

#### 2.1.4 首个实现为什么选择 DROID

原计划建议优先做 MolmoSpaces 完整视频试验；实际执行时数据仍在传输，已有可用的 DROID tracks、camera 和官方 annotations，因此先完成 DROID 数值链路。DROID RGB 需要另行取得上游视频并按官方映射重建，当前试产物明确 `frames_present=false`。

当前 writer 先逐条读取和检查 shape，再创建分片 mmap，并在写入阶段再次读取源记录。数组内存按记录控制，但全局候选和索引仍驻留内存。这是 pilot 的实现方式；尚不能宣称已完成百万级流式索引或最优构建吞吐。

实现保留 DROID 原有 `(T,N,*)` 对应关系、相机坐标 3D 和独立 `valid_3d`，同时存 measured K 与缩放后的 K。外参优先 `vggt_extrinsics`，否则使用 `optimized_extrinsics`，记录来源。这里只验证源值保持；两条校验路径共享几何假设，独立相机重投影和 RGB 对齐仍需补充。

目前可运行 CLI 是 `inspect-droid`、`build-droid`、`verify`、`benchmark`。独立 `relocation-test`、`export`、通用多子集 `build` 仍为计划能力。迁移测试本轮通过人工脚本执行。

### 2.2 转存后吞吐优化效果

#### 2.2.1 测试范围与方法

2026-09-14 历史输入快照记录：DROID 可用文件 5 个，共 2,340,248,092 B（约 2.18 GiB）；官方 train 22,363、test 1,169，共 23,532 clips。本次按清单顺序取前 512 条完整有效记录，均属 train，包含 769 个 motion ranges；并非从整个 1M 语料分层随机选出的代表样本。

512 条输出分为 2 个分片，每片 256 条，累计 4,750,097 个 `(帧,点)` 数值行。

基准在这一集合内，使用种子 `20260914` 生成相同的随机 sample ID 和起始帧请求，warmup 32 次，计时 512 次。sample ID 有放回采样，因此 512 次请求不代表每条记录恰好读取一次。每次最多读取 8 帧 × 32 点，点位置以相同的等距索引选择，两个路径都构造 NumPy 数组并计算相同字段的 checksum。

计时不包含 reader 初始化和初始索引加载。原始 reader 复用 tar 句柄并缓存已解析相机数据；缓存 reader 一次加载 Parquet 索引并复用 mmap。两边 checksum 均为 `97232661.30586773`。

#### 2.2.2 读取性能实测

| 指标 | 原始 tar + NPZ + JSON reader | NPY mmap + Parquet 索引 reader |
| --- | ---: | ---: |
| 512 次请求总耗时 | 2.89298 s | 0.05563 s |
| 吞吐 | 176.98 samples/s | 9,204.25 samples/s |
| P50 延迟 | 5.582 ms | 0.108 ms |
| P95 延迟 | 7.606 ms | 0.118 ms |

吞吐比为 **52.01 倍**；P95 延迟约降至原来的 1/64。主要原因是原始路径每次仍需读取并解压成对 NPZ 的数组，缓存路径可以直接访问窗口所需数值。JSON 数值预计算也减少工作，但基准已缓存相机解析结果，不能将全部加速归因于“每次重新解析 JSON”。

这是 **单进程、warm-cache 的数值 reader 微基准**。它未测量冷盘、索引启动耗时、多 worker、共享存储争用、视频帧解码、GPU 拷贝或完整训练。尚不能据此说训练快 52 倍，也不能把该倍数推广到其余子集；模型、图像或网络成为瓶颈后，端到端收益通常会变化。

没有单独测量峰值 RSS、CPU 利用率、系统调用次数和构建耗时分布。本日志没有重新跑基准，保留历史实测值及其条件。

#### 2.2.3 转存前后空间

| 同一 512 条 scope 的内容 | 逻辑字节 | 约 MiB |
| --- | ---: | ---: |
| 1,024 个 NPZ tar 成员 | 80,270,988 | 76.55 |
| 去重后的相机 JSON 成员 | 850,631 | 0.81 |
| 三份全局官方 annotations JSON | 17,099,292 | 16.31 |
| 所选逻辑源输入闭包 | 98,220,911 | 93.67 |
| mmap 试产物全部文件 | 104,659,916 | 99.81 |

输出相对上述输入闭包增加 **6.6%**。若仅与选中 NPZ 和相机成员比较，不计全局 annotations，则增加约 **29.0%**。这是压缩容器转可直接随机访问数组的空间成本。

源闭包包括构建索引用的 split 和来源清单涉及的全局标注，并不意味着输出另行逐字节保存了三份 JSON。所有容量数均为逻辑文件/成员字节，未测文件系统实际分配块数。不能用包含全部 23,532 clips 的 2.18 GiB 源容器对比 512 条输出计算“压缩率”，也不能将小试中的全局 JSON 开销简单线性外推。

#### 2.2.4 正确性与迁移检查

- 独立工具单元测试 2/2 通过；构建期检查 24 条，另行复验 64 条。按代码实际逻辑，记录检查采用清单等间距抽样，历史报告的“随机抽检”表述不够准确。
- 逐字段核对 2D、3D、两个 mask、两种 K 和外参，要求 shape、dtype、值一致；NaN 按相等处理。该结果覆盖抽检记录，不等于全量逐记录语义核验。
- SHA256SUMS 中 24 个交付文件全部通过；另加清单与 `PILOT_READY.json`，共 26 个文件。清单 SHA256 为 `292b4b98c1750322f96a423deccfd8abd4d930777cfc314192ea746335fd139e`。
- 历史迁移测试将包复制到另一根目录，不访问源数据即可加载 512 条索引并读取 `(8,32,3)` 窗口，校验和通过。它证明同机换路径可读；尚未证明金山云端实际可读，也未完整记录源目录权限隔离、成品只读挂载等更严格验收条件。

当前 source manifest 记录相对路径、大小、mtime；尚没有源内容 SHA256 和完整代码/配置构建指纹。输出 SHA256 用于完整性校验，不能替代输入已经传输完毕的证明。只匹配最终 `.tar` 文件名能排除常见 `.part`，仍需传输端完成清单和接收校验来保证文件不再写入。

证据：[原始试验报告](MOLMO_MOTION_DROID_PILOT_REPORT.md)、[benchmark JSON](molmo_motion_droid_pilot_benchmark.json)、[verify JSON](molmo_motion_droid_pilot_verify.json)、[source snapshot JSON](molmo_motion_droid_source_snapshot.json)、[reader/基准代码](../tools/molmo_motion_cache/src/molmo_motion_cache/cli.py)。

## 3. 腾讯云转存后迁移到金山云

采用 NumPy/Parquet/JSON 等通用格式、相对运行路径和配套 reader 后，缓存字节可以跨云复制使用；金山云无需重解原始 NPZ。这个结论针对新 motion cache 的设计与同机换路径验证。旧 LBM `.mmap` 的部分缓存标识包含绝对路径和 mtime，不能默认旧缓存换路径也无需重建。

建议交付顺序：完成选定 scope → 输出校验 → 复制到独立 incoming 目录 → 接收端验 SHA256 和完成标记 → 在金山云运行匹配版本 reader → 发布给训练使用。当前完成标记是 `PILOT_READY.json`，不能升级解释成完整 1M 的 `READY.json`。

迁移时同时交付代码或 wheel、环境依赖信息以及 schema 说明。常规读取不需要源文件；需要与源值对照的 `verify --checks N` 则需要原始数据。当前 CLI 可用 `--checks 0 --verify-hashes` 做无源内容读取的校验，但仍要求传入 `--source-root` 参数，这一点应在完善迁移 CLI 时改进。

本轮历史网络检查中，服务器间直连 SSH 未成功，源端也曾缺少 rsync。本机 SSH 别名不会自动存在于 tc_dev。正式传输前需要刷新网络状态并选定可达路径；可以通过两端可访问的中转位置传同一份不可变分片。目前没有执行金山云成品回传验收。

早期容量清点记录的金山云普通文件总量约 **235.21 GiB**（排除 `.cache`），发布清单约 **265.99 GiB**，当时缺少部分子集约 **30.78 GiB**。这些是历史计划中的输入量估算，并非今天刷新后的库存；额外上游视频和重建资产不在其中。历史 `du` 约 996G 的口径曾受挂载目录统计影响，传输预算应回到普通文件清单计算。

不能承诺全量成品只需回传 21–22 GiB：那个旧估算仅对应 MolmoSpaces 机器人子集的一种 MP4/Parquet 方案，不代表完整运动缓存。全量空间需要按子集、分辨率和轨迹密度分别试验后外推。

## 4. Git 仓库与开发过程治理

用户要求更换到 [cunningckn/lbm](https://github.com/cunningckn/lbm)，保留旧提交，并以更新基线整理历史。上一轮已完成迁移，保留现有仓库历史，没有重新初始化成空项目。

| 引用（本日志新增提交之前） | 提交 | 作用 |
| --- | --- | --- |
| `origin/main` | `a4fe469` | 新仓库默认主线，继承旧上游已取得的 master |
| `feature/tc_covert_adapt` | `d7d9149` | 当前开发分支，保留用户指定的 covert 拼写 |
| `archive/feature-tc-covert-adapt-pre-rebase-20260914` | `284d496` | 保存原 6 个提交，已推送新仓库 |
| tc_dev 命名 stash | `71b56e9a0b4b2c19774a10c5ad62f106c1af4bda` | `archive/pre-rebase-working-tree-20260914`，保存原未提交现场，仅在服务器本地 |

整理前的分支相对旧上游缓存主线为领先 6、落后 105 个提交；整理后在 `a4fe469` 上形成 5 个按职责划分的 Conventional Commits：

| 提交 | 主题 |
| --- | --- |
| `1906f61` | `feat(data): add InternData, MolmoAct, and RoboCOIN adapters` |
| `bbc4912` | `feat(data): add tc_dev conversion workflow` |
| `008b559` | `feat(cloud): add reproducible Kingsoft job submission` |
| `915714b` | `feat(data): add portable MolmoMotion mmap cache` |
| `d7d9149` | `docs(data): record MolmoMotion DROID pilot results` |

最新上游已有自动 adapter 注册机制，旧静态 DATASETS 列表提交不再适用；会回退 FK-cache 且配置链路不完整的独立 `state_kind` 改动未带入整理分支，保留在 archive。`out.log`、旧手工预构建选择和过时报告保留在 stash，未混入发布提交。原提交的 SHWplus 作者身份在相关整理提交中保留；其他新增工作标记 Codex，没有修改全局身份配置。

tc_dev 访问 GitHub 超时，上一轮通过本机转运 Git bundle 推送，远端三条引用与预期哈希一致。旧 `AoqunJin/lbm` 地址从本机返回 Repository not found，因此只能确认以服务器此前已获取的 `a4fe469` 为基线，不能声称已核实不可访问旧仓库此刻的最新状态。

另一个值得保留的能力是 `scripts/ks_submit.py`：提交云任务时为代码制作可追溯快照，并将云凭据排除在版本控制之外。它与跨云数据格式是独立问题，本轮没有据此宣称已完成生产云任务运行。

后续维护约定是以新仓库 main 为起点，每个提交只包含一个可说明的变化，记录对应验证；重写已共享历史前保存归档与工作现场。本次核查发现 CONTRIBUTING 和 CI push 分支仍写 master，应单独对齐新 main。本文记录该待办，不将文档整理扩展为 CI 行为修改。

## 5. 实施差距、后续顺序与复现入口

### 5.1 与完整计划相比还缺什么

| 能力 | 当前状态 | 后续验收方向 |
| --- | --- | --- |
| DROID 数值与相机转换 | 512 条 pilot 通过 | 完整到达清单、逐条失败清单、扩大真实样本覆盖 |
| 多子集 adapter | 仅 DROID | 分别核实 MolmoSpaces、EgoDex、HD-EPIC、Xperience、YT-VIS、Stereo4D schema |
| RGB 帧包和几何对齐 | 未实现 | 补上游视频，帧索引、2D 点、K 与图像联合验证 |
| 时间和对象完整契约 | 部分实现 | 当前区间保留 `end_frame_inclusive`；缺 FPS 时实现默认 15，尚需严格缺失策略和独立 SCHEMA 文档 |
| 分片规模与大数据内存 | pilot 每片 256 条、串行写入 | 按字节控制分片，批量索引，资源与峰值内存实测 |
| 中断恢复 | staging 和拒绝覆盖已实现 | 指纹校验、完整分片复用、真实中断/resume 测试 |
| 来源追溯与完整性 | 输出 hash、源 stat 清单 | 源 hash、代码/config/schema 指纹、原始 JSON 归档、完成标记强制校验 |
| 通用 reader/CLI | DROID 专用 reader | 缺模态和版本拒绝规则、显式 point ID 查询、export/relocation CLI |
| 金山云迁移 | 同机换路径测试通过 | 实际跨云传输、只读读取、源不可访问下验证 |
| 训练收益 | 数值 reader 热缓存微基准 | 冷/热、多 worker、RGB、GPU 等待、端到端 steps/s |
| 文档与环境 | README、试验报告、独立 pyproject | SCHEMA.md、MIGRATION.md、依赖锁定/离线包 |

`build-droid` 不传 `--limit` 只表示处理当前发现的完整有效输入；缺失或异常记录可能跳过并计数，不能据此宣称整个 DROID 或 MolmoMotion-1M 已全量完成。当前也尚未提供逐条异常原因清单和 resume。

建议先补输入交付校验、schema/失败清单和源指纹，再扩大 DROID 数值验证；并行准备具备完整视频的子集。之后加入帧包及真实训练适配，重测吞吐与容量，再执行金山云接收验证。全量转换应在这些接口和恢复规则稳定后推进。

### 5.2 复验现有试产物

以下命令在 tc_dev 中验证现有包，不覆盖历史报告或重新构建已存在输出：

```bash
cd /home/tione/workspace/kainingchen/lbm
export PYTHONPATH="$PWD/tools/molmo_motion_cache/src"
PY=/home/tione/workspace/kainingchen/xprobot_flow/.venv/bin/python
RAW=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m
OUT=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1/pilots/droid-pilot-512

"$PY" -m molmo_motion_cache verify \
  --source-root "$RAW" --output "$OUT" --checks 64 --verify-hashes

"$PY" -m molmo_motion_cache benchmark \
  --source-root "$RAW" --output "$OUT" \
  --samples 512 --warmup 32 --frames 8 --points 32 --seed 20260914
```

新建试验时，`build-droid --limit 512 --shard-size 256 --checks 24` 还需传入 source-root 和一个不存在的 output。不要把现有 pilot 目录当成可以覆盖的目标。

仅读取成品的示例，不需要原始数据目录：

```python
from molmo_motion_cache.reader import MMapDroidReader

reader = MMapDroidReader("/path/to/copied/droid-pilot-512")
sample_id = reader.sample_ids[0]
window = reader.get_window(sample_id, start=0, frames=8, points=32)
print(window["points3d"].shape)
```

代码入口：[独立工具 README](../tools/molmo_motion_cache/README.md)、[DROID builder](../tools/molmo_motion_cache/src/molmo_motion_cache/droid.py)、[reader](../tools/molmo_motion_cache/src/molmo_motion_cache/reader.py)、[CLI 与基准](../tools/molmo_motion_cache/src/molmo_motion_cache/cli.py)。历史完整计划保存在会话工作区的 `MOLMO_MOTION_TRANSFER_PLAN.md`（v1.1）；其中“尚未实施”的状态属于当时快照，当前完成范围以本文和试验产物为准。
