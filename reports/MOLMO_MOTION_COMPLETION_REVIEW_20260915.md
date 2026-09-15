# MolmoMotion 补全复核与实施结果

日期：2026-09-15。基线 0f51dba；分支 codex/molmo-motion-completion。
代码提交：bf43915（发布/续跑）、f217e8d（RGB及独立验收）。

## 结论

原 v1 是 annotations-only 数值缓存及已发布资产集合，不能表述为全模态训练数据完成。
本次完成发布/续跑约束加固、独立导出、小样本真实跨云验收、四段视频 RGB 实验、
独立数值字段比对和图像 0/1/4 worker 配对测试。旧 v1、原工作目录及失败现场均未覆盖。

全模态发布仍未完成：缺失上游模态需要原始数据/权限；已发布 MolmoSpaces 视频也存在
标注时钟与编码时钟偏差。没有提交全量 RGB 高资源任务，没有重新构建 339 GB 旧成品。

## 代码与校验

- 严格校验清单路径、重复项、摘要、符号链接及文件覆盖；公共 reader 校验 READY、版本和索引。
- 续跑绑定实际源内容、代码、schema 与参数；旧无契约成品可以只读审计/导出，不再凭 READY 跳过。
- 新根清单绑定组件 READY/清单和根元数据；明确 manifest-only 未验证数值/视频内容。
- 导出只复制显式正式文件，独立实体副本，拒绝覆盖，staging 校验后发布。
- 13 项自动化测试通过，包括损坏/遗漏/额外文件、非法路径/摘要、符号链接、参数变化、
  未就绪读取、旧格式续跑拒绝、根清单和导出排除审计目录。
- 公共真实数据读取证明合法旧 v1 仍可用。新整套全量 build 尚未再次执行，不能把单测称为全量发布验收。

工作代码位于腾讯云 `/home/tione/workspace/kainingchen/lbm-completion-20260915/tools/molmo_motion_cache`。
运行 `PYTHONPATH=src python3 -m unittest discover -s tests -v`。
新增使用说明见工具目录 SCHEMA.md 和 MIGRATION.md。

## 实际跨云验收

512 条 DROID 小样本、26 个文件、约 100 MiB，从腾讯云复制至金山云独立目录。
两端隔离进程均完成哈希前后复验和 8×32 数值窗口读取；原始 Tencent 路径被禁止访问。
金山云目录为 `/mnt/kpfs/workspace/jinaoqun/Projects/lbm-completion-tests-20260915`。
完整发布包跨云传输未执行；本次结果只能证明已测试小样本的跨云独立读取。

## RGB：实际发现了时间偏差

抽取 house_0 两个 clip 的 exo_camera_1 和 wrist_camera，共四段视频，长度分别 279/253 帧。
原尺寸 352×624，MP4 从 tar offset 直接读取，PNG/JPEG 连续写入二进制文件。
逐帧 PNG 解码与原始 RGB24 像素完全相等；所有视频帧数等于对应轨迹帧数。

标注 FPS=15，视频 time_base=1/19392、PTS 步长1280，实际约15.15 FPS。
末帧编码时间相对标注时间少0.1663–0.1835秒。默认 strict-time 拦截发布，并保留诊断 staging。
随后显式 frame-index 实验只按帧号关联，同时保留原始 PTS 和标注时间；
`time_semantics_verified=false`。未静默重采样或裁帧，未宣称时间语义已验证。

实验产物：
`/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/completion-tests/rgb-frame-index-20260915`

| 四段视频合计 | 字节 | 相对原 MP4 |
| --- | ---: | ---: |
| 原 MP4 成员 | 2,249,175 | 1× |
| PNG 帧数据 | 177,373,745 | 78.86× |
| JPEG q95 帧数据 | 57,046,658 | 25.36× |

JPEG PSNR 41.46–44.22 dB。构建进程峰值 RSS 约521 MiB，包含加载全量 clips/assets 索引；
不是每帧额外内存。每段流式处理约10.7–16.3秒。数据来自同一 house，不能代表全量压缩比。

全量设计建议以2 GiB独占写 shard、4 worker作为下一阶段受控扩样起点，先解决时间语义，
再扩大到多个 house/动作并测量容量；当前一视频一pilot shard，尚未实现2 GiB生产打包。
PNG作为无损参照；JPEG q95作为有损候选，不直接以四段结果决定全量格式或资源。

## 图像配对读取

每段12次请求，每次8个固定随机帧，四段共48次；分别测0、1、4 worker。
以下吞吐包含进程启动/初始化；JSON另存每个batch初始化、稳态耗时、延迟和峰值RSS。

| worker | MP4 请求/秒 | PNG 请求/秒 | JPEG 请求/秒 |
| --- | ---: | ---: | ---: |
| 0 | 8.31 | 23.00 | 61.13 |
| 1 | 8.10 | 22.26 | 57.27 |
| 4 | 22.58 | 69.33 | 145.64 |

MP4与PNG逐请求图像哈希一致。JPEG有损，因此其吞吐不构成同画质加速比。
这里无训练/GPU开销，样本小，系统page cache未受控；未清共享缓存、未声称冷盘性能。
原有11–29×只对应预热数值窗口，不能外推图像或端到端训练。

### 数值窗口：计入启动成本

DROID另测固定128个8帧×32点请求，0/1/4 worker，各worker复用reader。
原始/缓存逐请求字节哈希相同。生产reader初始化包含清单和索引校验，约4–5秒/worker。

| worker | raw请求/秒（含启动） | mmap请求/秒（含启动） |
| --- | ---: | ---: |
| 0 | 19.54 | 29.74 |
| 1 | 19.56 | 28.56 |
| 4 | 18.85 | 23.82 |

稳态p50约raw 3.84–4.11ms、mmap 0.07–0.08ms；128个请求太少，启动时间主导整段吞吐。
不能用稳态p50倒数冒充整体samples/s，也不能据此认定更多worker一定更快。
以 `NUMERIC_PERSISTENT_BENCHMARK.json` 为正式结果；早期每batch重建reader的诊断结果
`NUMERIC_BENCHMARK.json`保留作对照，不应作为持久worker性能。

## 独立数值语义

五个通用子集各抽每个split/kind前两条，直接读取源NPZ并独立整理字段；
不调用 builder 的 `_load_sample` 或 `_track_object`。共享部分仅为样本发现和tar成员读取。
完整数组比较涵盖首尾帧、float32转换、mask、Xperience置信度和keep_mask，
并检查相机原始值及稀疏帧索引；全部通过。原dtype见 SEMANTIC_CHECK.json。
DROID另取train/test首尾样本，直接源NPZ/JSON比较轨迹、mask、内外参及缩放结果，全部通过。
这证明所抽样本的字段一致性；未实施全量逐元素语义对照或跨源投影几何认证。

## 容量口径修正

按固定snapshot的82个源文件及发布清单普通文件统计，排除下载partial、jobs和pilots，
不含目录自身元数据占用：

| 范围 | 逻辑字节 | 已分配字节 |
| --- | ---: | ---: |
| 固定源快照 | 285,601,107,377 | 285,602,177,024 |
| 正式组件文件 | 338,592,848,267 | 338,642,755,584 |

逻辑容量增长约18.55%。同机hardlink去重后增量259,849,117,696字节。
旧报告294 GB源目录口径含额外文件；其15.19%不适合作为固定快照比较。
旧du目录总量还包含目录元数据，因此与普通文件stat汇总存在小差异。

## 下一阶段门槛

1. 确认15/15.15 FPS差异的上游语义，确定正式帧时间映射。
2. 扩大真实RGB样本覆盖，确定JPEG质量、空间预算和生产shard组织。
3. 获取缺失上游模态与许可访问，逐项运行原始重建流程。
4. 新完整发布及跨云全量迁移另行按清单验收；当前未发生。
5. 实际训练框架中做包含相机/图像/数值同步的端到端DataLoader吞吐测试。

未通过的计划项继续保留未勾选，尤其不能将RGB帧号实验写成全模态完成。

## 复现实验命令

在腾讯云工具目录下，设置 `PYTHONPATH=src`。以下路径缩写仅用于命令：

```bash
RAW=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m
OLD=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1
TEST=/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/completion-tests
export PYTHONPATH=src
python3 -m unittest discover -s tests -v
python3 semantic_check.py "$RAW" "$OLD" "$TEST/SEMANTIC_CHECK.json"
python3 droid_semantic_check.py "$RAW" "$OLD" "$TEST/DROID_SEMANTIC_CHECK.json"
python3 storage_check.py "$RAW" "$OLD" "$TEST/STORAGE_CHECK.json"
# 使用新的pilot目的地；更改代码或参数后旧指纹会拒绝续跑。
python3 -m molmo_motion_cache.rgb_pilot "$OLD" "$TEST/rgb-new-pilot" --alignment frame-index
python3 -m molmo_motion_cache.rgb_benchmark "$TEST/rgb-new-pilot" "$OLD/assets"
python3 selection_check.py "$OLD" "$TEST/rgb-new-pilot"
python3 -m molmo_motion_cache.numeric_benchmark "$RAW" "$OLD/subsets/droid" "$TEST/NUMERIC_PERSISTENT_BENCHMARK.json"
```

源码静态检查使用仓库锁定的Ruff 0.16.5；RGB依赖的实验环境为PyAV 17.0.1/Pillow 12.2.0。
工具目录Ruff、compileall和git diff --check通过；全仓Ruff另有8项既有问题，位于
scripts/ks_submit.py、scripts/tc_cloud/submit_molmo_motion_full.py及
interndata_a1/molmoact/robocoin适配器，本轮未改这些文件，不能宣称全仓lint通过。
可用独立包的 `[rgb]` extra安装RGB依赖。不会强制模型/训练框架接入这些读取接口。
